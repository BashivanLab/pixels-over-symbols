"""The random-encoder Natural model must see the features of its own encoder.

Feature caches are keyed only by backbone name, so the pretrained ResNet-CLIP cache
(data/features) and the cache of the seed-2025 random draw
(data/features/randinit_seed2025) have the same file names, and handing the wrong
one to a model raises no error. These tests pin the three things that must agree
on real HvM images:

1. the random-encoder feature cache,
2. the saved random backbone run on the raw images, and
3. the trained random-encoder model (seed 0), rebuilt with that backbone.

``test_wrong_cache_is_detected`` feeds the pretrained cache through the same model
and asserts the tolerances below reject it, so a passing suite cannot be an
artifact of loose bounds. Each test skips when its data is missing.
"""

import unittest

import torch

from models.model import CNNRNN
from models.vision_model import get_pretrained_stats, load_vision_backbone
from utils.const import OUTPUT_SIZE, TASK_NAME_TASK_INDEX
from utils.helper import load_repair_checkpoint
from utils.paths import CHECKPOINT_DIR, FEATURE_DIR, HVM_DIR
from utils.stim_io import HvMMetaData, pil_read_fn

BACKBONE = 'resnet50_clip.openai'
SEED = 2025
CACHE_DIR = FEATURE_DIR / f'randinit_seed{SEED}'
BACKBONE_SD = CACHE_DIR / f'{BACKBONE}_randinit_seed{SEED}.pth'
CHECKPOINT = (CHECKPOINT_DIR / 'natural' / f'{BACKBONE}_randinit_hs128_lstm_lr3e-4_rep0_wd1e-4_nomturk'
              / 'final_CNNRNN_model_rep0.pth')

# The cache was computed on a GPU in float16 autocast and stored in float16, so it
# cannot match a float32 CPU forward pass bit for bit: features differ by up to
# ~3e-4 (values up to ~0.4) and logits by up to ~1e-2, below these bounds and far
# below what the wrong cache produces (features ~25, logits ~2.6; see the last test).
FEATURE_TOL = 2e-3
LOGIT_TOL = 5e-2
N_TRIALS = 8
STIM_FRAMES = (0, 2, 5)  # stimulus frames of each 6-frame trial; the others are blank
TASK_ID = TASK_NAME_TASK_INDEX['interdms_ABAB_position_position']


def load_cache(folder):
    features = torch.load(folder / f'{BACKBONE}_train_features.pt', mmap=True)
    blank = torch.load(folder / f'{BACKBONE}_train_blank.pt')
    return features, blank


def tokens(feat):
    """(N, C, H, W) ResNet feature maps -> (N, HW, C) tokens, as CNNRNN.forward does."""
    return feat.flatten(2).transpose(1, 2)


@unittest.skipUnless(BACKBONE_SD.is_file() and (CACHE_DIR / f'{BACKBONE}_train_features.pt').is_file()
                     and (HVM_DIR / 'python').is_dir(),
                     f'needs the random-encoder cache in {CACHE_DIR} and the HvM images')
class TestRandomEncoderCache(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        meta = HvMMetaData(HVM_DIR)
        filenames = meta.df['filename'].astype(str).to_numpy()
        # stimuli spread over the whole metadata, i.e. over the three variation levels
        n_stim = N_TRIALS * len(STIM_FRAMES)
        picks = torch.linspace(0, len(filenames) - 1, n_stim).long().tolist()
        cls.rows = [meta.filename_to_idx[filenames[i]] for i in picks]
        read = pil_read_fn(dataset_stats=get_pretrained_stats(BACKBONE))
        cls.images = torch.stack([read(filenames[i]) for i in picks])

        cls.encoder = load_vision_backbone(BACKBONE, pretrained=False, state_dict_path=str(BACKBONE_SD))
        cls.cache, cls.blank = load_cache(CACHE_DIR)

        cls.task_index = torch.zeros(N_TRIALS, len(TASK_NAME_TASK_INDEX))
        cls.task_index[:, TASK_ID - 1] = 1

    def batches(self, cache, blank):
        """The same trials as raw images and as cached features."""
        T = 6
        img_batch = torch.zeros(N_TRIALS, T, *self.images.shape[1:])
        emb_batch = tokens(blank.float()).expand(N_TRIALS, T, -1, -1).clone()
        stim = iter(range(len(self.rows)))
        for b in range(N_TRIALS):
            for t in STIM_FRAMES:
                i = next(stim)
                img_batch[b, t] = self.images[i]
                emb_batch[b, t] = tokens(cache[self.rows[i]][None].float())[0]
        return img_batch, emb_batch

    def load_model(self):
        if not CHECKPOINT.is_file():
            self.skipTest(f'no trained random-encoder model at {CHECKPOINT}')
        model = CNNRNN(hidden_size=128, output_size=OUTPUT_SIZE, label_size=len(TASK_NAME_TASK_INDEX),
                       cnn_type=BACKBONE, seq_model_type='lstm', num_heads=8, pretrained=False,
                       vision_state_dict_path=str(BACKBONE_SD))
        state_dict = load_repair_checkpoint(CHECKPOINT)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # checkpoints may leave out the frozen backbone, which then comes from the saved draw
        self.assertEqual(unexpected, [])
        self.assertTrue(all(k.startswith('vision_backbone.') for k in missing), missing)
        return model.eval(), state_dict

    def test_cache_is_the_encoder_output(self):
        with torch.no_grad():
            feats = self.encoder.forward_features(self.images)
            blank = self.encoder.forward_features(torch.zeros(1, *self.images.shape[1:]))
        torch.testing.assert_close(self.cache[self.rows].float(), feats, rtol=0, atol=FEATURE_TOL)
        torch.testing.assert_close(self.blank.float(), blank, rtol=0, atol=FEATURE_TOL)

    def test_checkpoint_backbone_is_the_saved_draw(self):
        _, state_dict = self.load_model()
        backbone = {k[len('vision_backbone.'):]: v for k, v in state_dict.items()
                    if k.startswith('vision_backbone.')}
        if not backbone:
            self.skipTest('the checkpoint does not store the backbone')
        saved = torch.load(BACKBONE_SD, map_location='cpu')
        self.assertEqual(set(saved), set(backbone))
        for name, tensor in saved.items():
            self.assertTrue(torch.equal(backbone[name], tensor), name)

    def test_cached_and_raw_inputs_agree(self):
        model, _ = self.load_model()
        img_batch, emb_batch = self.batches(self.cache, self.blank)
        with torch.no_grad():
            out_cached, _, act_cached = model(emb_batch, self.task_index, get_act=True)
            out_raw, _, act_raw = model(img_batch, self.task_index, get_act=True)
        torch.testing.assert_close(out_cached, out_raw, rtol=0, atol=LOGIT_TOL)
        self.assertTrue(torch.equal(out_cached.argmax(-1), out_raw.argmax(-1)))
        for name in ('pre', 'post'):
            for a, b in zip(act_cached[name], act_raw[name]):
                torch.testing.assert_close(torch.as_tensor(a), torch.as_tensor(b), rtol=0, atol=LOGIT_TOL)

    def test_wrong_cache_is_detected(self):
        if not (FEATURE_DIR / f'{BACKBONE}_train_features.pt').is_file():
            self.skipTest(f'no pretrained cache in {FEATURE_DIR}')
        model, _ = self.load_model()
        wrong_cache, wrong_blank = load_cache(FEATURE_DIR)
        _, emb_batch = self.batches(self.cache, self.blank)
        _, wrong_batch = self.batches(wrong_cache, wrong_blank)
        self.assertGreater((wrong_batch - emb_batch).abs().max().item(), 10 * FEATURE_TOL)
        with torch.no_grad():
            out_right, _, _ = model(emb_batch, self.task_index)
            out_wrong, _, _ = model(wrong_batch, self.task_index)
        self.assertGreater((out_right - out_wrong).abs().max().item(), 10 * LOGIT_TOL)


if __name__ == '__main__':
    unittest.main()
