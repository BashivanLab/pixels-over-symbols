import unittest

import numpy as np
import pandas as pd
import torch

from analysis.eval_loop import eval_mturk, mean_over_draws, trial_noise_seeds
from analysis.experiment_pipeline import cache_stem
from analysis.factory import AB_CONFIG_DICT, DIM_MATCH_CONFIG_DICT, init_ab
from mturk.stimuli import sample_df
from mturk.task_configs import validate_task_configs
from utils.paths import HUMAN_DIR, HVM_DIR, TRIAL_DIR
from utils.stim_io import PRETRAINED_STATS


class TestCacheNames(unittest.TestCase):
    def test_cache_stem(self):
        # the names of the released evaluation files
        self.assertEqual(cache_stem('AB_LSTM_NOMTURK', AB_CONFIG_DICT['AB_LSTM_NOMTURK']), 'AB_LSTM_NOMTURK_match0.75')
        self.assertEqual(cache_stem('AB_LSTM_NOMTURK_U80', AB_CONFIG_DICT['AB_LSTM_NOMTURK_U80']),
                         'AB_LSTM_NOMTURK_U80_match0.8')
        self.assertEqual(cache_stem('AB_LSTM_NOMTURK_FINAL', AB_CONFIG_DICT['AB_LSTM_NOMTURK_FINAL']),
                         'AB_LSTM_NOMTURK_FINAL')
        self.assertEqual(cache_stem('DIM_MATCH_LSTM_NONOISE_NOMTURK',
                                    DIM_MATCH_CONFIG_DICT['DIM_MATCH_LSTM_NONOISE_NOMTURK']),
                         'DIM_MATCH_LSTM_NONOISE_NOMTURK_match0.75')


class TestNoiseDraws(unittest.TestCase):
    def test_trial_noise_seeds(self):
        seeds = trial_noise_seeds(model_seed=0, task_id=1, n_trials=100, draw=0)
        self.assertEqual(seeds, trial_noise_seeds(0, 1, 100, 0))
        self.assertEqual(len(set(seeds)), 100)
        for other in [trial_noise_seeds(1, 1, 100, 0), trial_noise_seeds(0, 2, 100, 0), trial_noise_seeds(0, 1, 100, 1)]:
            self.assertFalse(set(seeds) & set(other))

    def test_mean_over_draws(self):
        def pair_df(acc, conf):
            return pd.DataFrame({'pair_key': [(0, 1), (2, 3)], 'acc': [np.array([acc]), np.array([1.0])],
                                 'confs': [np.array([conf]), np.array([0.9])], 'mean_acc': [acc, 1.0],
                                 'mean_conf': [conf, 0.9]})

        out = mean_over_draws([pair_df(0.0, 0.2), pair_df(1.0, 0.6)])
        self.assertEqual([a[0] for a in out['acc']], [0.5, 1.0])
        np.testing.assert_allclose([c[0] for c in out['confs']], [0.4, 0.9])
        np.testing.assert_allclose(out['mean_acc'], [0.5, 1.0])
        other = pair_df(1.0, 0.6)
        other['pair_key'] = [(2, 3), (0, 1)]
        with self.assertRaises(AssertionError):
            mean_over_draws([pair_df(0.0, 0.2), other])


@unittest.skipUnless((HVM_DIR / 'python').is_dir() and (TRIAL_DIR / 'stim_subset.csv').is_file() and HUMAN_DIR.is_dir(),
                     'needs the HvM metadata and the MTurk data')
class TestEvalMturk(unittest.TestCase):
    def test_one_task(self):
        task_configs = validate_task_configs()
        img_loader, meta_data = sample_df(HVM_DIR, 16, 100, 3, df_path=task_configs[1].task_meta_csv,
                                          pretrained_stats=PRETRAINED_STATS['clip_resnet'])
        _, task_pair_dfs, _, task_features = eval_mturk(
            nat_models={}, ab_models={'lstm': [init_ab('LSTM')]}, vlm_models=[], task_ids=[41],
            img_loader=img_loader, meta_data=meta_data, task_configs=task_configs, device=torch.device('cpu'),
            get_act=True)
        self.assertEqual(list(task_pair_dfs), [41])
        humans, models = task_pair_dfs[41]['human'], task_pair_dfs[41]['lstm']
        self.assertEqual((len(humans), len(models)), (20, 1))
        for df in humans + models:
            self.assertEqual(len(df), 100)
            self.assertEqual(sorted(df['pair_key']), sorted(humans[0]['pair_key']))
        pre, post = task_features[41]['lstm'][0]
        self.assertEqual(len(post), 6)
        self.assertEqual(post[0].shape, (100, 128))


if __name__ == '__main__':
    unittest.main()
