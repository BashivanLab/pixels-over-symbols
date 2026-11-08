"""Short training runs of every trainer on the CPU.

Each run trains on a few hundred trials, so these tests check that the trainers
run end to end, write the expected files and freeze what they should, not the
accuracy of the models. They need the HvM metadata (data/original); the Natural
runs also need the random-encoder feature cache (data/features/randinit_seed2025)
and, for the substitution, the pretrained ResNet-CLIP weights in the Hugging Face
cache.
"""
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import try_to_load_from_cache

from models.dim_match import build_projection
from models.model import CNNRNN, CustomRNN
from train_abstract import train_abstract
from train_abstract_dim_match import train_abstract_dim_match
from train_abstract_frozen import train_abstract as train_abstract_frozen
from train_natural import train_natural
from train_natural_frozen import train_natural as train_natural_frozen
from utils.const import INPUT_SIZE, OUTPUT_SIZE, TASK_NAME_TASK_INDEX
from utils.paths import FEATURE_DIR, HVM_DIR

HAS_HVM = (HVM_DIR / 'python').is_dir() and (HVM_DIR / 'HvM_with_discfade').is_dir()
BACKBONE = 'resnet50_clip.openai'
RAND_DIR = FEATURE_DIR / 'randinit_seed2025'
RAND_SD = RAND_DIR / f'{BACKBONE}_randinit_seed2025.pth'
HIDDEN = 32
# 20 batches per epoch, so the mid-epoch checks run after every batch
TINY = dict(batch_size=16, train_dataset_size=320, val_dataset_size=64, num_workers=1, pin_mem=False,
            num_epochs=2, warmup_epochs=0, weight_decay=1e-4, grad_norm=2.0, rep_index=0, print_freq=1,
            hidden_size=HIDDEN, seq_model_type='lstm', num_heads=8)


def files(folder):
    return sorted(p.name for p in Path(folder).glob('*.pth'))


def same(a, b, prefixes):
    return all(torch.equal(a[k], b[k]) for k in a if k.startswith(prefixes))


@unittest.skipUnless(HAS_HVM, f'HvM images and metadata not found in {HVM_DIR}')
class TestAbstract(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def test_run_writes_checkpoints_and_curves(self):
        train_abstract(**TINY, lr=3e-3, save_mid_epoch=True, results_dir=str(self.out / 'a'))
        names = files(self.out / 'a')
        self.assertIn('final_RNN_model_rep0.pth', names)
        self.assertIn('RNN_model_epoch_0.pth', names)
        self.assertIn('RNN_model_epoch_1.pth', names)
        self.assertEqual(np.load(self.out / 'a' / 'val_avg_acc_rep0.npy').shape, (2,))
        model = CustomRNN(input_size=INPUT_SIZE, hidden_size=HIDDEN, output_size=OUTPUT_SIZE,
                          seq_model_type='lstm')
        model.load_state_dict(torch.load(self.out / 'a' / 'final_RNN_model_rep0.pth'))

    def test_same_seed_same_weights(self):
        for run in ('a', 'b'):
            train_abstract(**TINY, lr=3e-3, results_dir=str(self.out / run))
        a = torch.load(self.out / 'a' / 'final_RNN_model_rep0.pth')
        b = torch.load(self.out / 'b' / 'final_RNN_model_rep0.pth')
        self.assertTrue(all(torch.equal(a[k], b[k]) for k in a))

    def test_save_until_acc_stops_without_final_checkpoint(self):
        # every check reaches 0, so the run stops at the first one
        train_abstract(**TINY, lr=3e-3, save_mid_epoch=True, save_until_acc=0.0, results_dir=str(self.out / 'a'))
        self.assertEqual([n for n in files(self.out / 'a') if not n.startswith('RNN_model_acc')], [])

    def test_nat_to_abs_keeps_the_frozen_core(self):
        donor = CNNRNN(hidden_size=HIDDEN, output_size=OUTPUT_SIZE, label_size=len(TASK_NAME_TASK_INDEX),
                       seq_model_type='lstm', num_heads=8, no_backbone=True, feat_dim=16, num_input_tokens=4)
        donor_path = self.out / 'donor.pth'
        torch.save(donor.state_dict(), donor_path)
        train_abstract_frozen(**TINY, lr=3e-3, seq_model_path=str(donor_path), freeze_seq_model=True,
                              freeze_fc_out=True, results_dir=str(self.out / 'a'))
        trained = torch.load(self.out / 'a' / 'final_RNN_model_rep0.pth')
        self.assertTrue(same(trained, donor.state_dict(), ('seq_model.', 'norm_out.', 'fc_out.')))


@unittest.skipUnless(HAS_HVM, f'HvM images and metadata not found in {HVM_DIR}')
class TestDimMatch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)
        self.kwargs = dict(TINY, lr=3e-4, num_tokens=4, upsample_dim=16, noise_scale=0.35,
                           noise_pose_k=0.5, label_smoothing=0.2)

    def test_native_and_substitution(self):
        train_abstract_dim_match(**self.kwargs, results_dir=str(self.out / 'native'))
        native = torch.load(self.out / 'native' / 'final_RNN_model_rep0.pth')
        self.assertTrue(torch.equal(native['projection'], build_projection(85, 4, 16, seed=2025)))

        # theta + W_in: the donor's core and input pathway stay frozen, only the head trains
        train_abstract_dim_match(**self.kwargs, seq_model_path=str(self.out / 'native' / 'final_RNN_model_rep0.pth'),
                                 freeze_seq_model=True, freeze_input_pathway=True,
                                 results_dir=str(self.out / 'win'))
        win = torch.load(self.out / 'win' / 'final_RNN_model_rep0.pth')
        frozen = tuple(f'core.{m}.' for m in ('seq_model', 'proj', 'attn_pool', 'fc_in', 'norm_in'))
        self.assertTrue(same(win, native, frozen))
        self.assertFalse(torch.equal(win['core.fc_out.weight'], native['core.fc_out.weight']))


@unittest.skipUnless(HAS_HVM and RAND_SD.is_file(), f'needs the HvM images and the cache in {RAND_DIR}')
class TestNatural(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name)

    def test_random_encoder_model(self):
        train_natural(**TINY, lr=3e-4, cnn_type=BACKBONE, pretrained=False, vision_state_dict_path=str(RAND_SD),
                      emb_cache_dir=str(RAND_DIR), results_dir=str(self.out / 'a'))
        self.assertEqual(files(self.out / 'a'),
                         ['CNNRNN_model_epoch_0.pth', 'CNNRNN_model_epoch_1.pth', 'final_CNNRNN_model_rep0.pth'])
        trained = torch.load(self.out / 'a' / 'final_CNNRNN_model_rep0.pth')
        saved = torch.load(RAND_SD, map_location='cpu')
        self.assertTrue(all(torch.equal(trained[f'vision_backbone.{k}'], v) for k, v in saved.items()))

    @unittest.skipUnless((FEATURE_DIR / f'{BACKBONE}_train_features.pt').is_file()
                         and try_to_load_from_cache(f'timm/{BACKBONE}', 'model.safetensors') is not None,
                         'needs the ResNet-CLIP feature cache and pretrained weights')
    def test_abs_to_nat_keeps_the_frozen_core(self):
        donor = CustomRNN(input_size=INPUT_SIZE, hidden_size=HIDDEN, output_size=OUTPUT_SIZE, seq_model_type='lstm')
        donor_path = self.out / 'donor.pth'
        torch.save(donor.state_dict(), donor_path)
        train_natural_frozen(**TINY, lr=3e-4, cnn_type=BACKBONE, seq_model_path=str(donor_path),
                             freeze_seq_model=True, freeze_fc_out=True, label_smoothing=0.2,
                             results_dir=str(self.out / 'a'))
        trained = torch.load(self.out / 'a' / 'final_CNNRNN_model_rep0.pth')
        self.assertTrue(same(trained, donor.state_dict(), ('seq_model.', 'norm_out.', 'fc_out.')))


if __name__ == '__main__':
    unittest.main()
