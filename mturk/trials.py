"""Model inputs and labels for the MTurk trials.

A trial list (data/mturk/trials/*_trials.csv) has one column per shown stimulus (``stim1..stimN``,
N = ``frames_per_trial``) and an answer column (``act<i>``, 'x' = match, 'b' = non-match) per graded
stimulus; the delay between stimuli is not part of the data. The models were trained on 6-frame
sequences in which the last stimulus is in the last frame and blank delay frames fill the gap, so
:func:`mturk_trial_to_batches` lays every trial out that way: stimuli 1..N-1 go to frames 0..N-2,
stimulus N to frame 5, and the frames in between are blank with label 2 (no response).
``mturk.responses.process_human_responses`` and ``mturk.vlm.vlm_loader._load_one_seed`` put the human
and VLM responses in the same frames, so every responder is scored on the same frames.
"""

import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import torch

from mturk.helper import get_task_info_tuples
from mturk.io import _subpath_after, localize_mturk_csv_filename, _load_image_with_fallback
from utils.stim_io import HvMImageLoader, HvMMetaData, encode_features


def cached_feat_to_tokens(feat):
    """Put one cached per-image feature into the ``(N_tokens, N_dim)`` token layout.

    ``CNNRNN.forward`` treats a 4-D input ``(B, T, N_tokens, N_dim)`` as precomputed features. ViT
    caches already store ``(N_tokens, N_dim)`` per image. CNN caches store ``(C, H, W)``, which is
    flattened to ``(H*W, C)``, the same tokens the model makes from a CNN feature map
    (``features.flatten(2).transpose(1, 2)``).
    """
    if feat.dim() == 3:
        return feat.flatten(1).transpose(0, 1)
    return feat


def mturk_trial_to_batches(
        trial_df: pd.DataFrame,
        img_loader: HvMImageLoader,
        meta_data: HvMMetaData,
        trial_len: int = 6,
        img_size: int = 224,
        action_map: Dict[str, int] = None,
        ignore_learning: bool = True,
        emb_cache: torch.Tensor = None,
        blank_cache: torch.Tensor = None,
        load_images: bool = True,
):
    """Inputs, labels and stimulus info of the trials of one trial list, in the 6-frame layout.

    Args:
        trial_df: trial list with ``session``, ``stim*`` and ``act*`` columns.
        img_loader: loader of the MTurk stimulus subset (``mturk.stimuli.sample_df``).
        meta_data: HvM metadata; its ``stim_dir`` holds the images.
        trial_len: frames per model trial.
        img_size: image size of the raw-image input.
        action_map: answer -> label, default ``{'x': 1, 'b': 0}``.
        ignore_learning: drop the practice trials (session 0).
        emb_cache: cached encoder features of every HvM image (``(5760, C, H, W)`` for a CNN,
            ``(5760, N_tokens, N_dim)`` for a ViT); if given, the input is these features.
        blank_cache: cached feature of a blank frame, ``(1, ...)``.
        load_images: without ``emb_cache``, whether to load the images; if not, the input is None.

    Returns:
        ``(inputs, labels, task_info, stim_indices)``: inputs ``(n_trials, trial_len, 3, img_size,
        img_size)`` images or ``(n_trials, trial_len, N_tokens, N_dim)`` features; labels
        ``(n_trials, trial_len)`` with 2 on frames without a response; per trial the
        ``{frame: (category, identity, position, view, size)}`` of its stimuli (None on blank frames)
        and their row indices in ``img_loader.df``.
    """
    if action_map is None:
        action_map = {'x': 1, 'b': 0}

    stim_df = localize_mturk_csv_filename(img_loader.df.copy())

    # 'Variation.../file.png' -> index into the feature cache
    filename_to_idx = {
        str(_subpath_after(
            Path(re.sub(r'(?<=/)_', '', k)),
            segment='HvM_with_discfade'
        )): v
        for k, v in img_loader._task_cache.filename_to_idx.items()
    }
    if ignore_learning:
        trial_df = trial_df[trial_df["session"] != 0].reset_index(drop=True)
    stim_cols = sorted(
        [c for c in trial_df.columns if c.startswith('stim')],
        key=lambda x: int(x.replace('stim', ''))
    )
    act_cols = sorted(
        [c for c in trial_df.columns if c.startswith('act')],
        key=lambda x: int(x.replace('act', ''))
    )

    assert len(stim_cols) >= 2, "Need at least 2 stimulus columns"

    # stimulus column -> frame: 1..n-1 to frames 0..n-2, the last one to the last frame
    n_stims = len(stim_cols)
    col_idx_map = {i + 1: i for i in range(n_stims - 1)}
    col_idx_map[n_stims] = trial_len - 1

    n_trials = len(trial_df)
    inputs = None
    if emb_cache is not None:
        # every frame starts as the blank frame's feature; expand over (n_trials, trial_len) rather than
        # repeat, which would multiply an existing dimension of a 3-D CNN feature instead of adding one
        blank_frame = cached_feat_to_tokens(blank_cache[0]).to(torch.float32)
        inputs = blank_frame.expand(n_trials, trial_len, *blank_frame.shape).clone()
    elif load_images:
        inputs = torch.zeros((n_trials, trial_len, 3, img_size, img_size), dtype=torch.float32)

    acts_batch = torch.full((n_trials, trial_len), 2, dtype=torch.long)
    task_info_list = list()
    indices_batch = list()

    for trial_idx, (_, trial) in enumerate(trial_df.iterrows()):
        stim_indices = list()
        task_info = {i: None for i in range(trial_len)}
        for stim_col in stim_cols:
            col_num = int(stim_col.replace('stim', ''))
            frame_pos = col_idx_map[col_num]

            img_path = Path(trial[stim_col])
            if emb_cache is not None:
                img_idx = filename_to_idx[str(_subpath_after(img_path, 'images'))]
                inputs[trial_idx, frame_pos] = cached_feat_to_tokens(emb_cache[img_idx]).float()
            elif load_images:
                inputs[trial_idx, frame_pos] = _load_image_with_fallback(img_path, img_loader, meta_data.stim_dir)

            act_col = f'act{col_num}'
            if act_col in act_cols and pd.notna(trial[act_col]):
                acts_batch[trial_idx, frame_pos] = action_map[trial[act_col]]

            img_fp = _subpath_after(Path(img_path), segment='images')
            row_idx, info_tuple = get_task_info_tuples(stim_df, img_fp)
            stim_indices.append(row_idx)
            task_info[frame_pos] = info_tuple

        task_info_list.append(task_info)
        indices_batch.append(stim_indices)
    return inputs, acts_batch, task_info_list, indices_batch


def task_info_to_one_hot(
        task_info: List[Dict[int, Tuple]],
        trial_len: int = 6,
        emb_size: int = 8 + 64 + 9 + 3 + 1,
):
    """One-hot abstract model input of a batch of trials (``task_info`` of :func:`mturk_trial_to_batches`).

    Blank frames (``None``) stay all-zero, as the delay frames of the abstract training data.
    """
    one_hot_emb = torch.zeros((len(task_info), trial_len, emb_size), dtype=torch.float)
    for trial_idx, info_dict in enumerate(task_info):
        for t in range(trial_len):
            info_tuple = info_dict[t]
            if info_tuple is None:
                continue

            one_hot_emb[trial_idx, t] = encode_features(
                category=info_tuple[0],
                identity=info_tuple[1],
                position=info_tuple[2],
                view=info_tuple[3],
                size=info_tuple[4],
                category_size=8,
                identity_size=8,
                position_size=9,
                view_size=3,
                size_dim=1
            )
    return one_hot_emb
