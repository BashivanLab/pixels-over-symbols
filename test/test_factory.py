import tempfile
import unittest
from pathlib import Path

import torch

from analysis.factory import (
    AB_CONFIG_DICT, NAT_CONFIG_DICT, DIM_MATCH_CONFIG_DICT, DIM_MATCH_EMBEDDING_SIZE, DIM_MATCH_N_CONT, NUM_HEADS,
    find_best_checkpoint, init_ab, init_dim_match, init_nat, load_weights, run_dir,
)
from analysis.transplant_configs import AB_CONFIGS, DM_CONFIGS, NAT_CONFIGS
from models.dim_match import DimMatchCNNRNN, build_projection
from models.model import CNNRNN
from utils.const import TASK_NAME_TASK_INDEX
from utils.paths import CHECKPOINT_DIR

CHECKPOINT_GROUPS = {'abstract', 'abstract_until080', 'natural', 'dim_match', 'abs_to_nat', 'nat_to_abs',
                     'nat_to_abs_until080'}


class TmpDirTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()


def _small_core(hidden=32, feat_dim=16, num_tokens=4):
    # as the trainers build it
    return CNNRNN(hidden_size=hidden, output_size=2, label_size=len(TASK_NAME_TASK_INDEX), seq_model_type='GRU',
                  dropout=0.3, num_heads=NUM_HEADS, no_backbone=True, feat_dim=feat_dim, num_input_tokens=num_tokens)


def _inputs(batch_size=2, seq_len=3, dims=(DIM_MATCH_EMBEDDING_SIZE,)):
    torch.manual_seed(0)
    task_index = torch.zeros(batch_size, len(TASK_NAME_TASK_INDEX))
    task_index[:, 0] = 1
    return torch.randn(batch_size, seq_len, *dims), task_index


class TestConfigs(unittest.TestCase):
    def test_configs_point_into_checkpoint_groups(self):
        configs = {**AB_CONFIG_DICT, **NAT_CONFIG_DICT, **DIM_MATCH_CONFIG_DICT, **AB_CONFIGS, **NAT_CONFIGS,
                   **DM_CONFIGS}
        for key, cfg in configs.items():
            folder = Path(cfg['folder'])
            self.assertEqual(folder.parent.parent, CHECKPOINT_DIR, key)
            self.assertIn(folder.parent.name, CHECKPOINT_GROUPS, key)
            self.assertEqual(run_dir(cfg, 3).name, f"{folder.name}_rep3{cfg.get('suffix', '')}")
            self.assertEqual(cfg['match_acc'], cfg['acc_match'] is not None, key)
            self.assertNotIn('rand', key, 'a _rand suffix marks a random-init twin')

    def test_matched_checkpoint_selection(self):
        self.assertEqual(AB_CONFIG_DICT['AB_LSTM_NOMTURK']['acc_match'], 0.75)
        self.assertNotIn('any_epoch', AB_CONFIG_DICT['AB_LSTM_NOMTURK'])
        self.assertEqual(AB_CONFIG_DICT['AB_LSTM_NOMTURK_U80']['acc_match'], 0.80)
        self.assertTrue(AB_CONFIG_DICT['AB_LSTM_NOMTURK_U80']['any_epoch'])
        self.assertFalse(AB_CONFIG_DICT['AB_LSTM_NOMTURK_FINAL']['match_acc'])
        self.assertEqual(AB_CONFIGS['REG_XFER_ABS_THETA_U80']['acc_match'], 0.80)
        self.assertEqual(Path(AB_CONFIGS['REG_XFER_ABS_THETA_U80']['folder']).parent.name, 'nat_to_abs_until080')


class TestRebuild(TmpDirTest):
    def test_dim_match_round_trip(self):
        # a DimMatchCNNRNN built as train_abstract_dim_match.py builds it, saved and rebuilt by init_dim_match;
        # noise_scale 0 makes the forward pass deterministic
        upsample_dim, num_tokens = 16, 4
        source = DimMatchCNNRNN(
            core=_small_core(feat_dim=upsample_dim, num_tokens=num_tokens),
            projection=build_projection(DIM_MATCH_EMBEDDING_SIZE, num_tokens, upsample_dim, seed=2025),
            n_cont=DIM_MATCH_N_CONT, noise_scale=0.0, noise_pose_k=0.0,
        ).eval()
        torch.save(source.state_dict(), self.tmp / 'model.pth')

        rebuilt = init_dim_match(rnn_type='GRU', hidden=32, upsample_dim=upsample_dim, num_tokens=num_tokens)
        rebuilt.load_state_dict(torch.load(self.tmp / 'model.pth'))
        self.assertTrue(torch.equal(source.projection, rebuilt.projection))
        # the head count is not in the state dict, so check the forward pass, not only the keys
        self.assertEqual(rebuilt.core.attn_pool.num_heads, NUM_HEADS)
        emb, task_index = _inputs()
        with torch.no_grad():
            self.assertTrue(torch.allclose(source(emb, task_index)[0], rebuilt(emb, task_index)[0], atol=1e-6))

    def test_natural_checkpoint_with_or_without_encoder(self):
        # a compiled natural model saves '_orig_mod.'-prefixed keys and its frozen encoder; the release
        # strips both; either checkpoint must load into the encoder-free model
        source = _small_core().eval()
        state = source.state_dict()
        full = {f'_orig_mod.{k}': v for k, v in state.items()}
        full['_orig_mod.vision_backbone.stem.weight'] = torch.zeros(3)
        cfg = {'folder': str(self.tmp / 'run'), 'file': 'final_CNNRNN_model', 'match_acc': False, 'acc_match': None}
        emb, task_index = _inputs(dims=(4, 16))
        for rep, ckpt in enumerate([full, state]):
            run_dir(cfg, rep).mkdir()
            torch.save(ckpt, run_dir(cfg, rep) / f'final_CNNRNN_model_rep{rep}.pth')
            model = init_nat(rnn_type='GRU', hidden=32, num_tokens=4, feat_dim=16)
            self.assertIsNone(model.vision_backbone)
            model.load_state_dict(load_weights(rep, cfg, drop_backbone=True))
            with torch.no_grad():
                self.assertTrue(torch.equal(source(emb, task_index)[0], model(emb, task_index)[0]))

    def test_eight_heads(self):
        self.assertEqual(NUM_HEADS, 8)
        for core in ['DECODER', 'BERT']:
            heads = {getattr(m, 'num_heads', None) or getattr(m, 'num_attention_heads', None)
                     for m in init_ab(rnn_type=core, hidden=128).modules()} - {None}
            self.assertEqual(heads, {NUM_HEADS}, core)
        self.assertEqual(init_nat('LSTM', 128, 49, 2048).attn_pool.num_heads, NUM_HEADS)
        self.assertEqual(init_dim_match('LSTM', 128, upsample_dim=16, num_tokens=4).core.attn_pool.num_heads,
                         NUM_HEADS)


class TestFindBestCheckpoint(TmpDirTest):
    """Checkpoint selection, which fails silently when it picks the wrong file."""

    def setUp(self):
        super().setUp()
        for name in ['final_RNN_model_rep0.pth',
                     'RNN_model_acc0.62_epoch_0_batch_600.pth',
                     'RNN_model_acc0.74_epoch_0_batch_1500.pth',
                     'RNN_model_acc0.7963_epoch_2_batch_300.pth']:
            (self.tmp / name).touch()

    def test_epoch0_match(self):
        self.assertEqual(find_best_checkpoint(self.tmp, 'final_RNN_model', seed=0, match_acc=True, acc_match=0.75).name,
                         'RNN_model_acc0.74_epoch_0_batch_1500.pth')
        # later epochs only with any_epoch
        self.assertEqual(find_best_checkpoint(self.tmp, 'final_RNN_model', seed=0, match_acc=True, acc_match=0.80).name,
                         'RNN_model_acc0.74_epoch_0_batch_1500.pth')
        self.assertEqual(find_best_checkpoint(self.tmp, 'final_RNN_model', seed=0, match_acc=True, acc_match=0.80,
                                              any_epoch=True).name,
                         'RNN_model_acc0.7963_epoch_2_batch_300.pth')

    def test_final(self):
        self.assertEqual(find_best_checkpoint(self.tmp, 'final_RNN_model', seed=0, match_acc=False).name,
                         'final_RNN_model_rep0.pth')
        with self.assertRaises(FileNotFoundError):
            find_best_checkpoint(self.tmp, 'final_RNN_model', seed=1, match_acc=False)


if __name__ == '__main__':
    unittest.main()
