"""Stimulus I/O: loading HvM images/metadata and encoding them as task embeddings.

Contains the HvM image loaders and caches, the train/val split, the encode/decode
routines that convert between stimulus feature tuples (category, identity,
position, view, size) and the model's embedding vectors, and the mappers that
turn generated trial embeddings back into image tensors (or pre-extracted feature
caches) at DataLoader time.
"""

import os
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, Tuple, Optional, Sequence, Union, Callable, List, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision.transforms import v2

from utils.paths import HVM_DIR, TRIAL_DIR
from utils.const import PRETRAINED_STATS  # noqa: F401 (imported from here by the evaluation code)


def _convert_image_to_rgb(image):
    return image.convert("RGB")


def pil_read_fn(img_size=224, dataset_stats=None):
    """Image reader for the backbone normalization ``dataset_stats`` (ImageNet by default).

    ImageNet statistics: resize to 232 (bilinear), center crop 224. CLIP statistics:
    resize to 224 (bicubic), center crop 224.
    """
    if dataset_stats is None:
        dataset_stats = ([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
    mean, std = dataset_stats
    if mean == [0.485, 0.456, 0.406] and std == [0.229, 0.224, 0.225]:
        # resnet trained on imagenet
        transform = v2.Compose([
            v2.Resize((232, 232), interpolation=v2.InterpolationMode.BILINEAR, antialias=True),
            v2.CenterCrop(img_size),
            _convert_image_to_rgb,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ])
    elif mean == [0.48145466, 0.4578275, 0.40821073] and std == [0.26862954, 0.26130258, 0.27577711]:
        # clip resnet
        transform = v2.Compose([
            v2.Resize(img_size, interpolation=v2.InterpolationMode.BICUBIC, antialias=True),
            v2.CenterCrop(img_size),
            _convert_image_to_rgb,
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean, std),
        ])
    else:
        raise ValueError('dataset_stats not recognized')

    def _read(p):
        im = Image.open(p)
        if im is None:
            raise ValueError(f"PIL failed to load image at: {p}")
        im = transform(im)
        return im.contiguous()

    return _read


def save_train_val_split_no_mturk(
        df: pd.DataFrame,
        out_dir=None,
        val_ratio=1 / 4,
        mturk_stim_set: pd.DataFrame = None,
        seed: int = 2025,
):
    """Split the HvM images into train and val, with the MTurk stimuli removed entirely.

    The 100 MTurk images are dropped from the dataset, and what remains is split
    image-wise, stratified within each ``(category_id, obj_id, var)`` cell. Every
    (category, object) pair therefore appears in train, so every identity column of
    the one-hot input receives gradient, while the MTurk stimuli stay unseen. Val
    measures generalization to unseen views of seen objects, not to unseen objects.
    With the default arguments this gives the paper's split (4,271 train and 1,389
    val images).

    Args:
        df: Full HvM metadata frame (all splits).
        out_dir: Directory to write ``train.csv``/``val.csv`` into.
        val_ratio: Fraction of each stratum assigned to val.
        mturk_stim_set: Frame whose ``_id`` values identify the stimuli to drop.
            Joining on ``_id`` is exact (100 of 100); the filename hash would
            over-match across variation levels.
        seed: Seed for the per-stratum shuffle, so the split is reproducible.

    Returns:
        ``(train_df, val_df)``.
    """
    df = df.copy()

    if mturk_stim_set is not None:
        mturk_ids = set(mturk_stim_set['_id'].values)
        n_before = len(df)
        df = df[~df['_id'].isin(mturk_ids)].reset_index(drop=True)
        n_dropped = n_before - len(df)
        if n_dropped != len(mturk_ids):
            raise RuntimeError(
                f"Expected to drop {len(mturk_ids)} MTurk stimuli by '_id', dropped "
                f"{n_dropped}. The join key no longer matches; check stim_subset.csv."
            )
        print(f"[no-mturk split] dropped {n_dropped} MTurk stimuli, {len(df)} remain")

    rng = np.random.default_rng(seed)
    df["split"] = "train"

    strata = ["category_id", "obj_id"]
    if "var" in df.columns:
        strata.append("var")

    for _, cell in df.groupby(strata):
        idx = cell.index.to_numpy()
        if len(idx) < 2:
            # too small to split; keep it in train so the object stays trained
            continue
        n_val = int(round(len(idx) * val_ratio))
        n_val = min(max(n_val, 1), len(idx) - 1)  # never empty a stratum's train side
        rng.shuffle(idx)
        df.loc[idx[:n_val], "split"] = "val"

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)

    train_pairs = set(map(tuple, train_df[["category_id", "obj_id"]].drop_duplicates().values))
    all_pairs = set(map(tuple, df[["category_id", "obj_id"]].drop_duplicates().values))
    missing = all_pairs - train_pairs
    if missing:
        raise RuntimeError(
            f"{len(missing)} (category, object) pairs are absent from train, so their "
            f"identity columns would still be untrained: {sorted(missing)[:5]}"
        )

    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        train_df.to_csv(os.path.join(out_dir, "train.csv"), index=False)
        val_df.to_csv(os.path.join(out_dir, "val.csv"), index=False)
    return train_df, val_df


@dataclass
class HvMTaskCache:
    # sizes
    grid_size: int
    category_size: int
    identity_size: int
    position_size: int
    view_size: int
    size_dim: int
    embedding_size: int

    # dataframe-derived
    df_idx: np.ndarray
    cat_1b: np.ndarray
    id_1b: np.ndarray
    pos_1b: np.ndarray
    views: np.ndarray
    sizes: np.ndarray
    vars: np.ndarray
    filenames: np.ndarray

    # maps
    valid_triples: set
    triple_to_indices: Dict[Tuple[int, int, int], np.ndarray]
    cat_to_ids: Dict[int, np.ndarray]
    catid_to_positions: Dict[Tuple[int, int], np.ndarray]

    # images cache
    images: torch.Tensor = None  # [N, C, H, W]
    filename_to_idx: Dict[str, int] = None  # path -> row index


class HvMMetaData:
    """
    Metadata (and optionally the image tensors) of every HvM image, in the row
    order of python/hvmmeta_var*.csv; shared by the train and val loaders.
    """

    def __init__(
            self,
            root_dir: Union[str, Path],
            normalize: str = "minmax",
            state_dict: Optional[Dict[str, Any]] = None,
            grid_size=3,
    ):
        self.root_dir = Path(root_dir)
        self.grid_size = grid_size
        self.meta_dir = self.root_dir / "python"
        self.stim_dir = self.root_dir / "HvM_with_discfade"
        self.normalize = normalize
        self.filtered = False
        if state_dict is not None:
            print(f'[HvMMetaData] Loading MetaData state dict with {state_dict}')
            self.state = state_dict
        else:
            self.state = dict()

        self.df: Optional[pd.DataFrame] = None
        self.images = None

        # metadata filename_to_idx is the absolute index from the full csv with length 5760
        self.filename_to_idx = None
        self.load_metadata()
        self.filter_files()

    def load_metadata(self):
        """
         Load and optionally normalize metadata from the variation .csv files.
         Builds the category/object id mappings, grid positions and filename_to_idx.
         """
        meta_files, meta_vars = self.validate_structure()

        all_dfs = dict()
        for var, fp in zip(meta_vars, meta_files):
            df = load_meta_df(fp)

            for col in df.select_dtypes(include=['object']).columns:
                df[col] = df[col].map(lambda v: v.decode("utf-8") if isinstance(v, bytes) else v)

            if 'filename' in df.columns:
                df['filename'] = df['filename'].map(partial(localize_meta_filename, fp))

            df['var'] = var
            all_dfs[var] = df

        self.df = pd.concat(all_dfs.values(), ignore_index=True)
        if 'category' not in self.df.columns or 'obj' not in self.df.columns:
            raise ValueError("Metadata must contain 'category' and 'obj' columns")

        if self.normalize is not None:
            float_cols = ['rxy', 'rxz', 'ryz', 's']
            if 'norm_stats' not in self.state:
                self.state['norm_stats'] = dict()

            for col in float_cols:
                if self.normalize == 'minmax':  # Normalize all float columns to [-1, 1]
                    if col not in self.state['norm_stats']:
                        self.state['norm_stats'][col] = {'min': self.df[col].min(), 'max': self.df[col].max()}

                    s = self.state['norm_stats'][col]
                    if s['max'] > s['min']:
                        self.df[col] = 2 * (self.df[col] - s['min']) / (s['max'] - s['min']) - 1
                elif self.normalize == 'z-score':
                    if col not in self.state['norm_stats']:
                        self.state['norm_stats'][col] = {'mean': self.df[col].mean(), 'std': self.df[col].std()}

                    s = self.state['norm_stats'][col]
                    if s['std'] > 0:
                        self.df[col] = (self.df[col] - s['mean']) / s['std']
                else:
                    raise ValueError('normalize with minmax or z-score')

        self._build_mappings()
        self.df["cat_1b"] = self.df["category_id"].astype(int) + 1
        self.df["id_1b"] = self.df["obj_id"].astype(int) + 1
        pos, edges = self._build_discrete_positions(
            self.df, "ty", "tz", self.grid_size,
            existing_edges=self.state.get('grid_edges')
        )
        self.df["pos_1b"] = pos
        self.state['grid_edges'] = edges

        filenames = self.df["filename"].astype(str).to_numpy()
        self.filename_to_idx = {fn: i for i, fn in enumerate(filenames)}
        return self.df

    def get_state(self) -> Dict[str, Any]:
        """Returns the normalization stats and grid edges for saving."""
        return self.state

    def validate_structure(self):
        meta_files = sorted(self.meta_dir.glob('hvmmeta_var*.csv'))
        meta_vars = [int(f.stem.split('hvmmeta_var')[1]) for f in meta_files]

        variation_folders = sorted([
            f for f in self.stim_dir.iterdir()
            if f.is_dir() and f.name.startswith('Variation')
        ])
        folder_vars = [int(f.name.replace('Variation', '')[:2]) for f in variation_folders]

        if set(meta_vars) != set(folder_vars):
            raise ValueError(f"Mismatch between metadata files {meta_vars} and variation folders {folder_vars}")

        return meta_files, meta_vars

    def _build_mappings(self):
        """
        Build mappings from string category and obj labels to integer IDs.
        Each obj_id is unique *within* its category (i.e., 0, 1, 2,... per category).
        Adds new columns `category_id` and `obj_id` to self.df.
        """
        # Map categories to IDs
        unique_categories = sorted(self.df['category'].unique())
        self.category_to_id = {cat: i for i, cat in enumerate(unique_categories)}
        self.df['category_id'] = self.df['category'].map(self.category_to_id)

        self.obj_to_id = {}
        for cat, group in self.df.groupby('category'):
            objs_in_cat = sorted(group['obj'].unique())
            obj_map = {obj: i for i, obj in enumerate(objs_in_cat)}
            # Store mapping per category
            self.obj_to_id[cat] = obj_map

        self.df['obj_id'] = self.df.apply(
            lambda row: self.obj_to_id[row['category']][row['obj']], axis=1
        )
        return

    def check_file_existence(self):
        self.df['file_exists'] = self.df['filename'].apply(lambda p: Path(p).exists())

    def filter_files(self):
        """
        Filters self.df in-place to only include rows where files exist.
        """
        if self.df is None:
            self.load_metadata()
        else:
            if self.filtered:
                return self.df
        # Ensure file_exists column is present
        if 'file_exists' not in self.df.columns:
            self.check_file_existence()

        # Filter
        no_local_files = self.df[~self.df['file_exists']]
        print(f'removed {len(no_local_files)} rows from the original df')
        self.df = self.df[self.df['file_exists']].reset_index(drop=True)
        print(f'found {len(self.df)} local images after filtering')
        self.filtered = True
        return self.df

    def build_image_cache(
            self,
            img_read_fn: Callable,
            dtype: torch.dtype = torch.float32,
    ):
        # this image cache is loaded from all 5760 images
        imgs = []
        filenames = self.df["filename"].astype(str).to_numpy()
        for i, p in enumerate(filenames):
            im = img_read_fn(p)
            imgs.append(im)
        img_tensor = torch.stack(imgs, dim=0).to(dtype).contiguous()  # [N, H, W, C]
        self.images = img_tensor.share_memory_()
        return img_tensor

    @staticmethod
    def _build_discrete_positions(df, ty_col: str, tz_col: str, grid_size: int, existing_edges=None):
        tx = df[ty_col].to_numpy()
        ty = df[tz_col].to_numpy()

        if existing_edges:
            tx_edges = existing_edges['tx']
            ty_edges = existing_edges['ty']
        else:
            tx_edges = np.linspace(tx.min(), tx.max(), grid_size + 1)
            ty_edges = np.linspace(ty.min(), ty.max(), grid_size + 1)

        tx_bin = np.clip(np.digitize(tx, tx_edges[:-1], right=False), 1, grid_size)
        ty_bin = np.clip(np.digitize(ty, ty_edges[:-1], right=False), 1, grid_size)

        tx_bin_flipped = grid_size + 1 - tx_bin
        ty_bin_flipped = grid_size + 1 - ty_bin
        pos = (ty_bin_flipped - 1) * grid_size + tx_bin_flipped

        return pos.astype(int), {'tx': tx_edges, 'ty': ty_edges}


class HvMImageLoader:
    """
    used for reading images and initializing HvmTasks
    """

    def __init__(
            self,
            root_dir=None,
            metadata: Optional[HvMMetaData] = None,
            preload_images=True,
            img_size=224,
            df: pd.DataFrame = None,
            pretrained_stats=None
    ):
        """
        Args:
            root_dir (str or Path, optional):
                Root directory containing 'python/' and 'HvM_with_discfade/'.
                Defaults to data/original.
        """
        if root_dir is None:
            root_dir = HVM_DIR

        self.root_dir = Path(root_dir)
        self.preload_images = preload_images
        self.img_size = img_size
        self.pretrained_stats = pretrained_stats
        self.img_read_fn = pil_read_fn(
            self.img_size,
            self.pretrained_stats,
        )

        self.metadata = metadata if metadata is not None else HvMMetaData(root_dir=self.root_dir)
        if self.preload_images:
            if self.metadata.images is None:
                self.metadata.build_image_cache(self.img_read_fn)
        self.meta_dir = self.metadata.meta_dir
        self.stim_dir = self.metadata.stim_dir

        self.df = df if df is not None else self.metadata.df

    @classmethod
    def from_train_val(
            cls,
            root_dir: Union[str, Path] = None,
            train_vars=(0, 3, 6),
            val_vars=(0, 3, 6),
            preload_images: bool = True,
            grid_size: int = 3,
            img_size: int = 224,
            pretrained_stats=None,
    ) -> Tuple["HvMImageLoader", "HvMImageLoader"]:
        """
        Train and val loaders over the split in ``HvM_with_discfade/no_mturk/``.

        The split CSVs are written there by :func:`save_train_val_split_no_mturk`
        the first time they are missing (this needs the MTurk stimulus list,
        ``data/mturk/trials/stim_subset.csv``), and read back from disk on every
        call.
        """
        if root_dir is None:
            root_dir = HVM_DIR
        meta = HvMMetaData(root_dir, grid_size=grid_size)
        if preload_images:
            print('Using preloaded np img arrays')
            print(f'Using PIL image read function for caching with stats:')
            print('\t' + f'{pretrained_stats}')
            img_read_fn = pil_read_fn(img_size, pretrained_stats)
            meta.build_image_cache(img_read_fn=img_read_fn)

        stim_dir = meta.stim_dir
        split_dir = stim_dir / 'no_mturk'
        train_csv = split_dir / "train.csv"
        val_csv = split_dir / "val.csv"

        if not train_csv.exists() or not val_csv.exists():
            mturk_stim_set_fp = TRIAL_DIR / 'stim_subset.csv'
            if not mturk_stim_set_fp.exists():
                raise FileNotFoundError(
                    f'{mturk_stim_set_fp} is needed to build the train/val split '
                    f'(the MTurk stimuli are left out of both sets)')
            mturk_stim_set = pd.read_csv(mturk_stim_set_fp)
            print(f'Using mturk stim subset from {mturk_stim_set_fp} with {len(mturk_stim_set)} stimuli')
            train_df, val_df = save_train_val_split_no_mturk(
                meta.df,
                out_dir=split_dir,
                mturk_stim_set=mturk_stim_set,
            )
            print(f"Saved train.csv ({len(train_df)}) and val.csv ({len(val_df)}) to {split_dir}")
        print(f'Found train split csv at {train_csv}')
        print(f'Found validation split csv at {val_csv}')
        train_df = pd.read_csv(train_csv)
        val_df = pd.read_csv(val_csv)
        train_df['filename'] = train_df['filename'].map(partial(localize_csv_filename, stim_dir))
        val_df['filename'] = val_df['filename'].map(partial(localize_csv_filename, stim_dir))

        # instantiate loaders
        train_loader = cls(
            root_dir=root_dir,
            metadata=meta,
            preload_images=preload_images,
            df=train_df,
            img_size=img_size,
            pretrained_stats=pretrained_stats,
        )
        val_loader = cls(
            root_dir=root_dir,
            metadata=meta,
            preload_images=preload_images,
            df=val_df,
            img_size=img_size,
            pretrained_stats=pretrained_stats,
        )

        train_loader.prepare_for_tasks(
            grid_size,
            filter_vars=train_vars
        )
        val_loader.prepare_for_tasks(
            grid_size,
            filter_vars=val_vars
        )
        return train_loader, val_loader

    def load_image(self, path: str) -> torch.Tensor:
        """Load image either from memory cache or disk."""

        path = str(path)
        if hasattr(self, "_task_cache") and self._task_cache.images is not None:
            idx = self._task_cache.filename_to_idx[path]
            return self._task_cache.images[idx]
        else:
            if not Path(path).exists():
                raise FileNotFoundError(f"Image not found: {path}")
            image = self.read_img(path)
            return image

    def batch_load_image(self, paths):
        """
        find the image tensors from the provided paths
        """
        if hasattr(self, "_task_cache") and self._task_cache.images is not None:
            paths = [str(p) for p in paths]
            idxs = [self._task_cache.filename_to_idx[p] for p in paths]
            images = self._task_cache.images[idxs]
        else:
            images = list()
            for p in paths:
                images.append(self.load_image(p))
            images = torch.stack(images)
        return images

    def read_img(self, p):
        im = self.img_read_fn(p)
        return im.contiguous()

    def prepare_for_tasks(
            self,
            grid_size: int = 3,
            view_cols: Sequence[str] = ("rxy", "rxz", "ryz"),
            size_col: Optional[str] = "s",
            filter_vars=(0, 3, 6),
    ):
        """
        Build (and cache) everything HvMTaskDataset needs.
        If `preload_images=True`, the image tensor of the shared metadata cache
        (every HvM image, read once) is attached as well.
        """
        if hasattr(self, "_task_cache") and self._task_cache is not None:
            tc = self._task_cache
            return tc  # already prepared

        preload_images = self.preload_images
        task_df = self.df.copy()
        task_df = task_df.loc[task_df['var'].isin(filter_vars)]

        if not {"category_id", "obj_id", "ty", "tz"}.issubset(task_df.columns):
            raise ValueError("df must already have category_id, obj_id, ty, tz columns.")
        if not all(c in task_df.columns for c in view_cols):
            raise ValueError(f"df must already have {view_cols}")
        if not size_col in task_df.columns:
            raise ValueError(f"df must already have {size_col}")
        category_size = 8
        identity_size = 8
        position_size = grid_size * grid_size
        view_size = len(view_cols)
        size_dim = 1

        embedding_size = (
                category_size +
                category_size * identity_size +
                position_size +
                view_size +
                size_dim
        )

        # maps
        triple_to_indices = defaultdict(list)
        cat_to_ids = defaultdict(set)
        catid_to_positions = defaultdict(set)
        row_indices = np.arange(len(task_df))

        for i, r in task_df.iterrows():
            c, ident, p = int(r["cat_1b"]), int(r["id_1b"]), int(r["pos_1b"])
            triple_to_indices[(c, ident, p)].append(i)
            cat_to_ids[c].add(ident)
            catid_to_positions[(c, ident)].add(p)

        valid_triples = set(triple_to_indices.keys())
        triple_to_indices = {k: np.asarray(v, dtype=np.int64) for k, v in triple_to_indices.items()}
        cat_to_ids = {c: np.asarray(sorted(list(v)), dtype=np.int64) for c, v in cat_to_ids.items()}
        catid_to_positions = {
            k: np.asarray(sorted(list(v)), dtype=np.int64) for k, v in catid_to_positions.items()
        }

        # continuous
        views = task_df.loc[:, view_cols].to_numpy(dtype=np.float32) if view_size > 0 else None
        sizes = task_df[size_col].to_numpy(dtype=np.float32) if size_dim == 1 else None
        filenames = task_df["filename"].astype(str).to_numpy()
        vars = task_df["var"].to_numpy(dtype=np.int8)

        images_np = None
        filename_to_idx = self.metadata.filename_to_idx

        if preload_images:
            images_np = self.metadata.images
        self.df = task_df

        self._task_cache = HvMTaskCache(
            grid_size=grid_size,
            category_size=category_size,
            identity_size=identity_size,
            position_size=position_size,
            view_size=view_size,
            size_dim=size_dim,
            embedding_size=embedding_size,
            df_idx=row_indices,
            cat_1b=task_df["cat_1b"].to_numpy(dtype=np.int8),
            id_1b=task_df["id_1b"].to_numpy(dtype=np.int8),
            pos_1b=task_df["pos_1b"].to_numpy(dtype=np.int8),
            views=views,
            sizes=sizes,
            vars=vars,
            filenames=filenames,

            valid_triples=valid_triples,
            triple_to_indices=triple_to_indices,
            cat_to_ids=cat_to_ids,
            catid_to_positions=catid_to_positions,

            images=images_np,
            filename_to_idx=filename_to_idx
        )
        return


class HvMImageMapper:
    """
    given embeddings, returns image tensors.
    """

    def __init__(
            self,
            task_dataset=None,
            img_loader: HvMImageLoader = None,
            img_size=224
    ):
        self.img_size = img_size
        if task_dataset is not None:
            img_loader = task_dataset.hvm_loader
        else:
            if img_loader is None:
                raise ValueError("Either dataset or img_loader must be provided.")
            img_loader = img_loader

        if not hasattr(img_loader, "_task_cache") or img_loader._task_cache is None:
            raise ValueError("img_loader must be prepared for tasks before passing to HvMImageMapper")
        cache = img_loader._task_cache

        self.loader = img_loader
        self.cache = cache
        self.row_cache = dict()  # save compute for finding the index of the one-hot tuple

    def find(self, decode_tuple):
        cat, obj, pos, view, size = decode_tuple
        to_int = lambda x: x.item() if torch.is_tensor(x) else int(x)
        to_float = lambda x: x.item() if torch.is_tensor(x) else float(x)

        cat, obj, pos = map(to_int, (cat, obj, pos))
        size = to_float(size)
        view = tuple(to_float(e) for e in view)

        decode_tuple = (cat, obj, pos, view, size)
        if decode_tuple in self.row_cache:
            idxs = self.row_cache[decode_tuple]
            return int(np.random.choice(idxs))

        mask = (
                (self.cache.cat_1b == int(cat)) &
                (self.cache.id_1b == int(obj)) &
                (self.cache.pos_1b == int(pos))
        )
        if self.cache.views is not None:
            mask &= np.all(self.cache.views == np.array(view, dtype=self.cache.views.dtype), axis=1)
        if self.cache.sizes is not None:
            mask &= (self.cache.sizes == float(size))
        idxs = np.where(mask)[0]

        if idxs.size == 0:
            raise RuntimeError(f"No image for category {cat}, id {obj}, pos {pos}, view {view}, size {size}")
        # several images share a feature tuple only in var0; one is drawn at random
        idx = int(np.random.choice(idxs))
        self.row_cache[decode_tuple] = idxs
        return idx

    def _batch_decode_and_find(self, embs):
        # this maps the one-hot to filenames, specific to each split
        decode_tuples = batch_decode(
            embs,
            self.cache.category_size,
            self.cache.identity_size,
            self.cache.position_size,
            self.cache.view_size,
            self.cache.size_dim
        )
        files = list()
        for decode_tuple in zip(*decode_tuples):
            files.append(self.cache.filenames[self.find(decode_tuple)])
        return files, decode_tuples

    def _load_batch(self, paths):
        return self.loader.batch_load_image(paths)

    def map_sequence(self, emb_arr):
        """
        emb_arr:   array of shape (T, embed_dim)
        """
        N = emb_arr.shape[0]
        C, H, W = 3, self.img_size, self.img_size
        imgs_out = torch.zeros((N, C, H, W), dtype=torch.float32)
        task_info = {i: None for i in range(N)}

        zero_mask = torch.all(torch.isclose(
            emb_arr,
            torch.tensor(0.0, dtype=emb_arr.dtype)
        ), dim=1)
        nonzero_idx = torch.nonzero(~zero_mask).squeeze(1)
        assert not nonzero_idx.numel() == 0, 'this should not happen'

        subset = emb_arr[nonzero_idx]
        files, decode_tuples = self._batch_decode_and_find(subset)
        imgs_out[nonzero_idx] = self._load_batch(files)

        per_sample_decodes = list(zip(*decode_tuples))  # list of (cats, objs, poses, views, sizes)
        for j, flat_pos in enumerate(nonzero_idx.tolist()):
            task_info[flat_pos] = per_sample_decodes[j]
        return imgs_out, task_info

    def map_batch(self, emb_batch: np.ndarray):
        """
        emb_batch: array of shape (B, T, embed_dim)
        returns:   tensor of shape (B, T, 3, H, W)
        """
        B, T, D = emb_batch.shape
        flat = emb_batch.reshape(B * T, D)  # (B*T, D)
        imgs_flat, task_info_flat = self.map_sequence(flat)  # imgs_flat: (B*T, C, H, W)

        imgs = imgs_flat.view(B, T, *imgs_flat.shape[1:])
        task_info = {
            b: {
                t: task_info_flat[b * T + t]
                for t in range(T)
            }
            for b in range(B)
        }
        return imgs, task_info


class CachedHvMImageMapper(HvMImageMapper):
    """
    given embeddings, returns the cached backbone features of the images.

    ``embedding_cache_path`` holds ``forward_features`` of every HvM image, indexed
    by the image's row in the full metadata (``filename_to_idx``); blank frames get
    the features of an all-zero image from ``blank_cache_path``.
    """

    def __init__(
            self,
            embedding_cache_path: str,
            blank_cache_path: str,
            task_dataset=None,
            img_loader: HvMImageLoader = None,
            img_size=224,
    ):
        super().__init__(task_dataset, img_loader, img_size)
        self.emb_cache = torch.load(embedding_cache_path)
        self.blank_cache = torch.load(blank_cache_path)

        if len(self.emb_cache.shape) == 4:
            # (N, C, H, W) -> (N, C, H*W) -> (N, H*W, C)
            self.emb_cache = self.emb_cache.flatten(2).transpose(1, 2)
            self.blank_cache = self.blank_cache.flatten(2).transpose(1, 2)

        self.blank_cache = self.blank_cache.squeeze(0)

    def map_sequence(self, emb_arr):
        """
        emb_arr:   array of shape (T, embed_dim)
        """
        N = emb_arr.shape[0]
        L, D = self.blank_cache.shape

        embs_out = self.blank_cache.unsqueeze(0).repeat(N, 1, 1).to(torch.float32)
        task_info = {i: None for i in range(N)}

        zero_mask = torch.all(torch.isclose(
            emb_arr,
            torch.tensor(0.0, dtype=emb_arr.dtype)
        ), dim=1)
        nonzero_idx = torch.nonzero(~zero_mask).squeeze(1)
        assert not nonzero_idx.numel() == 0, 'this should not happen'

        subset = emb_arr[nonzero_idx]
        files, decode_tuples = self._batch_decode_and_find(subset)

        idxs = [self.cache.filename_to_idx[str(p)] for p in files]
        embs = self.emb_cache[idxs].float()  # Convert half -> float for training

        embs_out[nonzero_idx] = embs

        per_sample_decodes = list(zip(*decode_tuples))
        for j, flat_pos in enumerate(nonzero_idx.tolist()):
            task_info[flat_pos] = per_sample_decodes[j]
        return embs_out, task_info

    def map_batch(self, emb_batch: np.ndarray):
        """
        emb_batch: array of shape (B, T, embed_dim)
        returns:   tensor of shape (B, T, L, D)
        """
        B, T, D = emb_batch.shape
        flat = emb_batch.reshape(B * T, D)  # (B*T, D)
        img_embs_flat, task_info_flat = self.map_sequence(flat)  # img_emb_flat: (B*T, L, D)
        img_embs = img_embs_flat.view(B, T, *img_embs_flat.shape[1:])

        task_info = {
            b: {
                t: task_info_flat[b * T + t]
                for t in range(T)
            }
            for b in range(B)
        }
        return img_embs, task_info


def localize_meta_filename(meta_fp, val):
    local_base = meta_fp.parent.parent / 'HvM_with_discfade'
    local_fp = local_base / Path(*Path(val).parts[5:])
    return local_fp


def _subpath_after(p: Path, segment: str) -> Optional[Path]:
    """Return the subpath (as Path) after `segment`"""
    try:
        idx = p.parts.index(segment)
    except ValueError:
        return None
    return Path(*p.parts[idx + 1:])


def localize_csv_filename(
        stim_dir: Union[str, Path],
        val: Union[str, Path],
        segment: str = "HvM_with_discfade",
) -> Path:
    """
    Build a local path by taking the portion of `val` after `segment`
    and anchoring it under the occurrence of `segment` in `stim_dir`.
    Both paths must contain `segment`.
    """
    stim_dir = Path(stim_dir)
    val_p = Path(val)
    dir_idx = stim_dir.parts.index(segment)
    stim_base = Path(*stim_dir.parts[: dir_idx + 1])

    relative = _subpath_after(val_p, segment)
    return stim_base / relative


def load_meta_df(fp):
    return pd.read_csv(fp)


def decode_embedding(
        embedding: torch.Tensor,
        category_size: int,
        identity_size: int,
        position_size: int,
        view_size: int,
        size_dim: int
):
    """
        Decode an embedding vector back into feature values.
        """
    if embedding.ndim != 1:
        raise ValueError(f"Expected 1D embedding tensor, got shape {embedding.shape} instead")
    category_end = category_size
    identity_end = category_end + category_size * identity_size
    position_end = identity_end + position_size
    cont_start = position_end

    if embedding[:category_end].sum().item() > 1:
        raise RuntimeError("Invalid category one-hot (more than one hot)")
    category = torch.argmax(embedding[:category_end]).item()
    cat_1b = category + 1

    # Extract full identity slice first, then index
    full_identity = embedding[category_end:identity_end]
    identity_offset = category * identity_size
    identity_slice = full_identity[identity_offset:identity_offset + identity_size]
    if identity_slice.sum().item() > 1:
        raise RuntimeError("Invalid identity one-hot (more than one hot)")
    identity = torch.argmax(identity_slice).item()
    id_1b = identity + 1

    if embedding[identity_end:position_end].sum().item() > 1:
        raise RuntimeError("Invalid position one-hot (more than one hot)")
    position = torch.argmax(embedding[identity_end:position_end]).item()
    pos_1b = position + 1

    view = tuple(embedding[cont_start:cont_start + view_size].tolist())
    size = embedding[cont_start + view_size].item()
    return cat_1b, id_1b, pos_1b, view, size


def batch_decode(
        emb: torch.Tensor,
        category_size: int,
        identity_size: int,
        position_size: int,
        view_size: int,
        size_dim: int
):
    """
    return category of shape (N,), identity of shape (N,), position of shape (N,),
    views of shape (N, view_size), sizes of shape (N,)
    1-based indices for category, identity, position
    """
    N, D = emb.shape
    category_end = category_size
    identity_end = category_end + category_size * identity_size
    position_end = identity_end + position_size
    cont_start = position_end

    cats = emb[:, :category_end].argmax(dim=1)
    full_identity = emb[:, category_end:identity_end].reshape(N, category_size, identity_size)
    ids = full_identity[torch.arange(N, device=emb.device), cats].argmax(dim=1)  # (N,)
    pos = emb[:, identity_end:position_end].argmax(dim=1)  # (N,)
    views = emb[:, cont_start: cont_start + view_size]  # (N, view_size)
    sizes = emb[:, cont_start + view_size]  # (N,)

    cat_1b = (cats + 1)
    id_1b = (ids + 1)
    pos_1b = (pos + 1)
    return cat_1b, id_1b, pos_1b, views, sizes


def encode_features(
        category: int,
        identity: int,
        position: int,
        view: List[float],
        size: float,
        category_size: int,
        identity_size: int,
        position_size: int,
        view_size: int,
        size_dim: int
):
    """
    return one-hot encoded embedding tensor
    """
    embedding_size = (
            category_size +
            category_size * identity_size +
            position_size +
            view_size +
            size_dim
    )
    emb = torch.zeros(embedding_size)
    emb[category - 1] = 1

    # identity block
    identity_start = category_size
    identity_end = identity_start + category_size * identity_size
    emb[identity_start:identity_end].zero_()
    identity_offset = identity_start + (category - 1) * identity_size + identity - 1
    emb[identity_offset] = 1

    position_start = identity_end
    position_end = position_start + position_size
    emb[position_start:position_end].zero_()
    emb[position_start + position - 1] = 1

    start = embedding_size - (view_size + size_dim)
    if view_size > 0:
        emb[start:start + view_size] = torch.tensor(view, dtype=emb.dtype)
    if size_dim == 1:
        emb[start + view_size] = size
    return emb
