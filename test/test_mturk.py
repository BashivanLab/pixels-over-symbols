import unittest

import pandas as pd
import torch

from analysis.helper import HUMAN_TASK_IDS
from mturk.responses import process_human_responses
from mturk.stimuli import sample_df
from mturk.task_configs import validate_task_configs
from mturk.trials import cached_feat_to_tokens, task_info_to_one_hot
from mturk.vlm.vlm_loader import get_vlm_responses
from utils.paths import HVM_DIR, TRIAL_DIR, VLM_DIR


class TestFrameLayout(unittest.TestCase):
    def test_cached_feat_to_tokens(self):
        cnn = torch.arange(2 * 3 * 4.).reshape(2, 3, 4)  # (C, H, W)
        tokens = cached_feat_to_tokens(cnn)
        self.assertEqual(tokens.shape, (12, 2))
        self.assertTrue(torch.equal(tokens[5], cnn[:, 1, 1]))
        vit = torch.randn(197, 8)
        self.assertIs(cached_feat_to_tokens(vit), vit)

    def test_one_hot_blank_frames(self):
        info = (1, 2, 5, [0.1, -0.2, 0.3], 0.5)
        emb = task_info_to_one_hot([{0: info, 1: None, 2: None, 3: None, 4: None, 5: info}])
        self.assertEqual(emb.shape, (1, 6, 85))
        self.assertTrue(emb[0, 1:5].eq(0).all())
        self.assertTrue(torch.equal(emb[0, 0], emb[0, 5]))

    def test_human_responses_layout(self):
        # an InterDMS trial (4 stimuli) answered at stimuli 3 and 4: frames 2 and 5 of the 6-frame layout
        detail = pd.DataFrame({'trial_id': [1, 1], 'resp_index': [2, 3], 'response': ['x', float('nan')]})
        out = process_human_responses([detail], trial_len=4, pad_len=6)
        self.assertEqual(out.shape, (1, 1, 6, 3))
        self.assertEqual(out[0, 0].argmax(-1).tolist(), [2, 2, 1, 2, 2, 2])

    def test_sample_df_needs_the_subset(self):
        with self.assertRaises(FileNotFoundError):
            sample_df(HVM_DIR, 16, 100, 3, df_path=TRIAL_DIR / 'no_such_subset.csv')


@unittest.skipUnless((TRIAL_DIR / 'stim_subset.csv').is_file(), 'needs data/mturk/trials')
class TestTrialLists(unittest.TestCase):
    def test_mturk_tasks(self):
        self.assertEqual(sorted(validate_task_configs()), HUMAN_TASK_IDS)


@unittest.skipUnless(VLM_DIR.is_dir(), 'needs data/mturk/vlm')
class TestVLM(unittest.TestCase):
    def test_vlm_runs(self):
        runs = get_vlm_responses('interdms_ABAB_identity_identity', 'qwen3-vl-30b-a3b-thinking')
        self.assertEqual(len(runs), 6)
        for probs in runs:
            self.assertEqual(probs.shape, (100, 6, 3))
            self.assertTrue(probs.sum(-1).eq(1).all())
            # answers at stimuli 3 and 4 only, the last one in the last frame
            answered = probs[..., 2] == 0
            self.assertFalse(answered[:, [0, 1, 3, 4]].any())
            self.assertTrue(answered[:, [2, 5]].any(0).all())


if __name__ == '__main__':
    unittest.main()
