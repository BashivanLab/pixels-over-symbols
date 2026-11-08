"""DataLoader construction and batch collation for the task battery.

Provides the collate functions that turn embedding batches into model inputs for
the abstract, naturalistic, and cached-feature pipelines, and the two top-level
entry points :func:`prepare_abstract_dataloaders` and
:func:`prepare_natural_dataloaders`. Both train on all 43 tasks.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, default_collate

from utils.paths import FEATURE_DIR
from task.multi_task import MultiTaskDataset, TASK_IDX_TASK_TYPE  # noqa: F401 (TASK_IDX_TASK_TYPE is imported from here)
from utils.helper import worker_init_fn, find_emb_cache
from utils.stim_io import HvMImageLoader, HvMImageMapper, CachedHvMImageMapper


def abstract_collate(batch, task_dataset):
    """Collate embedding batches for the abstract model, decoding each frame to trial metadata."""
    inputs, labels, task_indices = zip(*batch)
    task_info = [
        {
            i: task_dataset.decode_embedding(emb) if not np.allclose(emb, 0) else None
            for i, emb in enumerate(batch)
        }
        for batch in inputs
    ]
    inputs = torch.stack(inputs, dim=0)
    labels = torch.stack(labels, dim=0)
    task_indices = torch.stack(task_indices, dim=0)
    return inputs, labels, task_indices, task_info


def natural_collate(batch, mapper: HvMImageMapper):
    """Collate a batch into raw image tensors ``(B, T, C, H, W)`` via the HvM image mapper."""
    # batch is list of (emb_arr, actions, task_idx)
    emb, actions, task_idx = zip(*batch)
    emb = torch.stack(emb)  # (B, T, D)
    imgs, task_info = mapper.map_batch(emb)  # img tensor shape (B, T, C, H, W)
    actions = default_collate(actions).long()  # torch tensor (B, T)
    task_idx = default_collate(task_idx).long()  # torch tensor (B,)
    return imgs, actions, task_idx, task_info


def cached_collate(batch, mapper: CachedHvMImageMapper):
    """Collate a batch into pre-extracted backbone features ``(BT, L, D)`` (frozen-backbone path)."""
    # batch is list of (emb_arr, actions, task_idx)
    emb, actions, task_idx = zip(*batch)
    emb = torch.stack(emb)  # (B, T, D)
    embs, task_info = mapper.map_batch(emb)  # img tensor shape (BT, L, D)
    actions = default_collate(actions).long()  # torch tensor (B, T)
    task_idx = default_collate(task_idx).long()  # torch tensor (B,)
    return embs, actions, task_idx, task_info


class NaturalCollateFn:
    """Picklable ``collate_fn`` wrapping :func:`natural_collate` with its image mapper."""

    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, batch):
        return natural_collate(batch, self.mapper)


class AbstractCollateFn:
    """Picklable ``collate_fn`` wrapping :func:`abstract_collate` with its dataset."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __call__(self, batch):
        return abstract_collate(batch, self.dataset)


class CachedCollateFn:
    """Picklable ``collate_fn`` wrapping :func:`cached_collate` with its cached-feature mapper."""

    def __init__(self, mapper):
        self.mapper = mapper

    def __call__(self, batch):
        return cached_collate(batch, self.mapper)


def prepare_abstract_dataloaders(
        root_dir=None,
        batch_size=128,
        train_dataset_size=4096,
        val_dataset_size=1024,
        num_workers=6,
        pin_mem=True,
        train_vars=(0, 3, 6),
        val_vars=(0, 3, 6),
):
    """Build train/val loaders that yield decoded embeddings for the abstract model.

    Returns ``(train_loader, val_loader, train_dataset, val_dataset)``. Images are
    not loaded; only the abstract feature embeddings are used. The split is the
    one of :func:`utils.stim_io.save_train_val_split_no_mturk`.
    """
    train_hvm_loader, val_hvm_loader = HvMImageLoader.from_train_val(
        root_dir=root_dir,
        preload_images=False,
        train_vars=train_vars,
        val_vars=val_vars,
    )
    train_dataset = MultiTaskDataset(
        hvm_loader=train_hvm_loader,
        dataset_size=train_dataset_size,
    )
    val_dataset = MultiTaskDataset(
        hvm_loader=val_hvm_loader,
        dataset_size=val_dataset_size,
        task_weights=train_dataset.task_weights  # keep distribution consistent
    )
    print(f"[Abstract] Train dataset size: {len(train_dataset)}")
    print(f"[Abstract] Validation dataset size: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=AbstractCollateFn(train_dataset),
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=AbstractCollateFn(val_dataset),
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=True,
        worker_init_fn=worker_init_fn,
    )
    print(f"[Abstract] Train dataset size after wrapping: {len(train_dataset)}")
    print(f"[Abstract] Validation dataset size after wrapping: {len(val_dataset)}")

    return train_loader, val_loader, train_dataset, val_dataset


def prepare_natural_dataloaders(
        root_dir=None,
        batch_size=128,
        train_dataset_size=4096,
        val_dataset_size=1024,
        num_workers=6,
        pin_mem=True,
        train_vars=(0, 3, 6),
        val_vars=(0, 3, 6),
        pretrained_stats=None,
        cnn_type=None,
        emb_cache_dir=None,
):
    """Build train/val loaders for the naturalistic model (raw images or cached features).

    If ``cnn_type`` has a feature cache in ``emb_cache_dir`` (default
    ``data/features``), the loaders yield pre-extracted backbone features (frozen
    backbone); otherwise they yield raw image tensors and the model runs its
    backbone. Returns ``(train_loader, val_loader, train_mapper, val_mapper)``.
    The cache is indexed by the image's row in the full metadata, so it serves
    the train and val sets alike.
    """
    train_hvm_loader, val_hvm_loader = HvMImageLoader.from_train_val(
        root_dir=root_dir,
        preload_images=True,
        train_vars=train_vars,
        val_vars=val_vars,
        pretrained_stats=pretrained_stats,
    )

    train_dataset = MultiTaskDataset(
        hvm_loader=train_hvm_loader,
        dataset_size=train_dataset_size,
    )
    val_dataset = MultiTaskDataset(
        hvm_loader=val_hvm_loader,
        dataset_size=val_dataset_size,
        task_weights=train_dataset.task_weights  # keep distribution consistent
    )

    find_cached = False
    if cnn_type is not None:
        paths = find_emb_cache(
            cnn_type,
            # the caller's emb_cache_dir must win: caches are keyed only by
            # backbone name, so a run whose features come from somewhere else
            # (e.g. a randomly-initialised encoder) would otherwise silently
            # train on the default pretrained cache for the same backbone
            emb_cache_dir=emb_cache_dir if emb_cache_dir is not None
            else FEATURE_DIR
        )
        if paths:
            find_cached = True
            cached_train_emb_path, cached_val_emb_path = paths[0], paths[1]
            blank_train_emb_path, blank_val_emb_path = paths[2], paths[3]

    if find_cached:
        print(f"[Natural] ---  Using Cached Features for {cnn_type} ---")
        train_mapper = CachedHvMImageMapper(
            task_dataset=train_dataset,
            embedding_cache_path=cached_train_emb_path,
            blank_cache_path=blank_train_emb_path,
        )
        val_mapper = CachedHvMImageMapper(
            task_dataset=val_dataset,
            embedding_cache_path=cached_val_emb_path,
            blank_cache_path=blank_val_emb_path,
        )
    else:
        train_mapper = HvMImageMapper(task_dataset=train_dataset)
        val_mapper = HvMImageMapper(task_dataset=val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=NaturalCollateFn(train_mapper) if not find_cached else CachedCollateFn(train_mapper),
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=pin_mem,
        worker_init_fn=worker_init_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=NaturalCollateFn(val_mapper) if not find_cached else CachedCollateFn(val_mapper),
        num_workers=num_workers,
        persistent_workers=True,
        pin_memory=pin_mem,
        worker_init_fn=worker_init_fn,
    )
    print(f"[Natural] Train dataset size after wrapping: {len(train_dataset)}")
    print(f"[Natural] Validation dataset size after wrapping: {len(val_dataset)}")

    return train_loader, val_loader, train_mapper, val_mapper
