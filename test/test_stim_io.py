"""HvM metadata, the train/val split and the stimulus embedding.

The tests that read the HvM images and metadata skip when data/original is missing.
"""
import tempfile
import unittest
from pathlib import Path

import pandas as pd
import torch

from utils.paths import HVM_DIR, TRIAL_DIR
from utils.stim_io import HvMImageLoader, HvMMetaData, decode_embedding, encode_features

HAS_HVM = (HVM_DIR / 'python').is_dir() and (HVM_DIR / 'HvM_with_discfade').is_dir()
SHIPPED_SPLIT = HVM_DIR / 'HvM_with_discfade' / 'no_mturk'
SIZES = dict(category_size=8, identity_size=8, position_size=9, view_size=3, size_dim=1)


class TestEmbedding(unittest.TestCase):
    def test_encode_decode_round_trip(self):
        for cat, ident, pos in [(1, 1, 1), (3, 7, 5), (8, 8, 9)]:
            view, size = (0.25, -0.5, 0.75), -0.125
            emb = encode_features(cat, ident, pos, list(view), size, **SIZES)
            self.assertEqual(emb.shape, (85,))
            self.assertEqual(emb[:81].sum().item(), 3)  # one-hot category, identity, position
            self.assertEqual(decode_embedding(emb, **SIZES), (cat, ident, pos, view, size))


@unittest.skipUnless(HAS_HVM, f'HvM images and metadata not found in {HVM_DIR}')
class TestMetadata(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = HvMMetaData(HVM_DIR)

    def test_every_image_present(self):
        # the feature caches are indexed by row of the full metadata
        self.assertEqual(len(self.meta.df), 5760)
        self.assertEqual(len(self.meta.filename_to_idx), 5760)

    def test_feature_ids(self):
        df = self.meta.df
        self.assertEqual(sorted(df['cat_1b'].unique()), list(range(1, 9)))
        self.assertEqual(df.groupby('cat_1b')['id_1b'].nunique().tolist(), [8] * 8)
        self.assertTrue(df['pos_1b'].between(1, 9).all())
        self.assertEqual(sorted(df['var'].unique()), [0, 3, 6])
        for col in ['rxy', 'rxz', 'ryz', 's']:
            self.assertGreaterEqual(df[col].min(), -1)
            self.assertLessEqual(df[col].max(), 1)


@unittest.skipUnless(HAS_HVM and (TRIAL_DIR / 'stim_subset.csv').is_file(),
                     'needs the HvM data and data/mturk/trials/stim_subset.csv')
class TestSplit(unittest.TestCase):
    """from_train_val writes the split on first use, next to the images, and reads it back."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        root = Path(cls.tmp.name)
        (root / 'python').symlink_to((HVM_DIR / 'python').resolve())
        (root / 'HvM_with_discfade').mkdir()
        for folder in (HVM_DIR / 'HvM_with_discfade').glob('Variation*'):
            (root / 'HvM_with_discfade' / folder.name).symlink_to(folder.resolve())
        cls.split_dir = root / 'HvM_with_discfade' / 'no_mturk'
        cls.train_loader, cls.val_loader = HvMImageLoader.from_train_val(root_dir=root, preload_images=False)
        cls.train = pd.read_csv(cls.split_dir / 'train.csv')
        cls.val = pd.read_csv(cls.split_dir / 'val.csv')
        cls.mturk_ids = set(pd.read_csv(TRIAL_DIR / 'stim_subset.csv')['_id'])

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_split_sizes(self):
        self.assertEqual((len(self.train), len(self.val)), (4271, 1389))
        self.assertEqual((len(self.train_loader.df), len(self.val_loader.df)), (4271, 1389))

    def test_mturk_stimuli_left_out(self):
        self.assertEqual(len(self.mturk_ids), 100)
        self.assertFalse(self.mturk_ids & set(self.train['_id']))
        self.assertFalse(self.mturk_ids & set(self.val['_id']))
        self.assertFalse(set(self.train['_id']) & set(self.val['_id']))

    def test_every_object_in_both_sets(self):
        for df in (self.train, self.val):
            self.assertEqual(len(df[['category_id', 'obj_id']].drop_duplicates()), 64)

    @unittest.skipUnless((SHIPPED_SPLIT / 'train.csv').is_file(), f'no split CSVs in {SHIPPED_SPLIT}')
    def test_matches_shipped_split(self):
        def strip(p):
            return p[p.index('HvM_with_discfade'):]

        for name, new in (('train.csv', self.train.copy()), ('val.csv', self.val.copy())):
            shipped = pd.read_csv(SHIPPED_SPLIT / name)
            for df in (shipped, new):
                df['filename'] = df['filename'].map(strip)
            pd.testing.assert_frame_equal(new, shipped)


if __name__ == '__main__':
    unittest.main()
