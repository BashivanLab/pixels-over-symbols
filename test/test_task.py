"""Trial generation of the 43 tasks on the HvM stimuli.

Every generated trial is decoded back into its stimuli and checked against the
task rule. Skips when the HvM metadata (data/original) is missing.
"""
import unittest

import numpy as np
import torch

from task.dataloader import prepare_abstract_dataloaders
from task.dms import CtxDMSDataset, DMSDataset, InterDMSDataset
from task.multi_task import ALL_TASKS, MultiTaskDataset, TASK_IDX_TASK_TYPE
from task.nback import NBackDataset
from utils.const import TASK_NAME_TASK_INDEX
from utils.paths import HVM_DIR
from utils.stim_io import HvMImageLoader

HAS_HVM = (HVM_DIR / 'python').is_dir() and (HVM_DIR / 'HvM_with_discfade').is_dir()
FEATURES = ['category', 'identity', 'position']


def same(feature, a, b):
    """Whether two decoded stimuli (category, identity, position, view, size) match on ``feature``."""
    if feature == 'category':
        return a[0] == b[0]
    if feature == 'identity':
        return a[:2] == b[:2]
    return a[2] == b[2]


def trials(ds):
    """Regenerate ``ds`` and yield its trials as (decoded frames, actions); blank frames decode to None."""
    ds.reset()
    for i in range(len(ds)):
        data, actions, _ = ds[i]
        frames = [None if torch.all(f == 0) else ds.decode_embedding(f) for f in data]
        yield frames, actions.tolist()


@unittest.skipUnless(HAS_HVM, f'HvM images and metadata not found in {HVM_DIR}')
class TestTasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loader = HvMImageLoader(preload_images=False)
        cls.loader.prepare_for_tasks(grid_size=3)

    def make(self, cls_, **kwargs):
        return cls_(hvm_loader=self.loader, dataset_size=50, pad_to=6, **kwargs)

    def test_registry_matches_task_ids(self):
        self.assertEqual(set(ALL_TASKS), set(TASK_NAME_TASK_INDEX))
        for name, (cls_, kwargs) in ALL_TASKS.items():
            task_id = TASK_NAME_TASK_INDEX[name]
            ds = cls_(hvm_loader=self.loader, dataset_size=1, pad_to=6, **kwargs)
            self.assertEqual(int(np.argmax(ds.task_index)) + 1, task_id, name)
            self.assertIs(TASK_IDX_TASK_TYPE[task_id], cls_, name)

    def test_dms(self):
        for feature in FEATURES:
            for frames, actions in trials(self.make(DMSDataset, feature=feature)):
                self.assertEqual(actions[:-1], [2] * 5)
                self.assertEqual([f is not None for f in frames], [True] + [False] * 4 + [True])
                self.assertEqual(actions[-1], int(same(feature, frames[0], frames[-1])))

    def test_nback(self):
        for n in (1, 2, 3):
            for feature in FEATURES:
                for frames, actions in trials(self.make(NBackDataset, feature=feature, nback_n=n)):
                    self.assertTrue(all(f is not None for f in frames))
                    self.assertEqual(actions[:n], [2] * n)
                    for j in range(n, 6):
                        self.assertEqual(actions[j], int(same(feature, frames[j - n], frames[j])))

    def test_interdms(self):
        # (sample, response) frame positions of problems A and B among the four stimulus frames
        pairs = {'AABB': ((0, 1), (2, 3)), 'ABAB': ((0, 2), (1, 3)), 'ABBA': ((0, 3), (1, 2))}
        for pattern, (pair_a, pair_b) in pairs.items():
            for f1 in FEATURES:
                for f2 in FEATURES:
                    ds = self.make(InterDMSDataset, pattern=pattern, feature_1=f1, feature_2=f2)
                    for frames, actions in trials(ds):
                        sp = [t for t, f in enumerate(frames) if f is not None]
                        self.assertEqual((len(sp), sp[0], sp[-1]), (4, 0, 5))
                        self.assertEqual(sorted(t for t, a in enumerate(actions) if a != 2),
                                         sorted([sp[pair_a[1]], sp[pair_b[1]]]))
                        for feature, (s, r) in ((f1, pair_a), (f2, pair_b)):
                            self.assertEqual(actions[sp[r]], int(same(feature, frames[sp[s]], frames[sp[r]])))

    def test_ctxdms(self):
        for features in (['position', 'category', 'identity'], ['position', 'identity', 'category'],
                         ['identity', 'position', 'category'], ['category', 'identity', 'position']):
            for frames, actions in trials(self.make(CtxDMSDataset, features=features)):
                sp = [t for t, f in enumerate(frames) if f is not None]
                self.assertEqual((len(sp), sp[0], sp[-1]), (3, 0, 5))
                self.assertEqual(actions[:-1], [2] * 5)
                # the outcome of the first comparison picks the feature of the second
                first, middle, last = (frames[t] for t in sp)
                feature = features[1] if same(features[0], first, middle) else features[2]
                self.assertEqual(actions[-1], int(same(feature, middle, last)))

    def test_multitask_sizes(self):
        ds = MultiTaskDataset(hvm_loader=self.loader, dataset_size=1000)
        sizes = {name: len(sub) for name, sub in ds.datasets.items()}
        self.assertEqual(sum(sizes.values()), 1000)
        # split evenly over the four task types, then over the tasks of each type
        for prefix, n_tasks in (('dms', 3), ('interdms', 27), ('ctxdms', 4)):
            group = [s for name, s in sizes.items() if name.split('_')[0] == prefix]
            self.assertEqual((len(group), sum(group)), (n_tasks, 250))
        self.assertEqual(sum(s for name, s in sizes.items() if 'back' in name), 250)

    def test_trials_resampled_every_epoch(self):
        ds = MultiTaskDataset(hvm_loader=self.loader, dataset_size=200)
        first, second = list(ds), list(ds)
        self.assertEqual(len(first), 200)
        emb, actions, task_index = first[0]
        self.assertEqual((tuple(emb.shape), tuple(actions.shape), tuple(task_index.shape)), ((6, 85), (6,), (43,)))
        self.assertFalse(all(torch.equal(a[0], b[0]) for a, b in zip(first, second)))


@unittest.skipUnless(HAS_HVM and (HVM_DIR / 'HvM_with_discfade' / 'no_mturk' / 'train.csv').is_file(),
                     'needs the HvM data and its train/val split')
class TestAbstractDataloader(unittest.TestCase):
    def test_batches(self):
        train_loader, val_loader, train_ds, val_ds = prepare_abstract_dataloaders(
            batch_size=8, train_dataset_size=64, val_dataset_size=32, num_workers=1, pin_mem=False)
        self.assertEqual((len(train_ds), len(val_ds)), (64, 32))
        inputs, labels, task_index, task_info = next(iter(train_loader))
        self.assertEqual(tuple(inputs.shape), (8, 6, 85))
        self.assertEqual(tuple(labels.shape), (8, 6))
        self.assertEqual(tuple(task_index.shape), (8, 43))
        self.assertEqual(len(task_info), 8)
        # task_info decodes every stimulus frame and leaves blank frames None
        for trial, info in zip(inputs, task_info):
            self.assertEqual([info[t] is None for t in range(6)], [bool(torch.all(f == 0)) for f in trial])


if __name__ == '__main__':
    unittest.main()
