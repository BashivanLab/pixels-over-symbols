"""VLM responses on the MTurk trials, in the format of the human responses.

Each VLM was queried six times on every trial list (``<model>--<run>/results_<task>_trials.csv`` under
the provider's folder in data/mturk/vlm); ``resp_<k>`` holds its answer at stimulus k. A run takes the
place of a participant in ``eval_mturk``.
"""

from pathlib import Path
from typing import List

import pandas as pd
import torch

from utils.paths import VLM_DIR

RESPONSE_MAP = {"b": 0, "x": 1}
# results folder of each VLM family
VLM_PROVIDER_DIRS = {'gemini': 'results-google', 'nemotron': 'results-nvidia', 'qwen': 'results-qwen'}


def vlm_results_dir(model_name: str) -> Path:
    for family, folder in VLM_PROVIDER_DIRS.items():
        if family in model_name:
            return VLM_DIR / folder
    raise ValueError(f'unknown VLM {model_name}')


def _load_one_seed(file_path: Path, pad_len: int, ignore_learning: bool) -> torch.Tensor:
    """One run as a ``(n_trials, pad_len, 3)`` one-hot tensor over ``[b, x, no response]``.

    The last stimulus goes to the last frame (``pad_len - 1``), the others to frame k - 1, as in
    ``mturk.trials.mturk_trial_to_batches``; frames without a response stay at "no response".
    """
    df = pd.read_csv(file_path)
    if ignore_learning:
        df = df[df["session"] > 0]
    df = df.sort_values("trial_index").reset_index(drop=True)

    resp_cols = sorted(
        [c for c in df.columns if c.startswith("resp_")],
        key=lambda x: int(x.split("_")[1]),
    )
    frame_numbers = [int(c.split("_")[1]) for c in resp_cols]

    probs = torch.zeros(len(df), pad_len, 3)
    probs[:, :, 2] = 1.0
    max_frame = max(frame_numbers)
    for col, frame_num in zip(resp_cols, frame_numbers):
        seq_pos = pad_len - 1 if frame_num == max_frame else frame_num - 1
        for trial_idx, raw_resp in enumerate(df[col]):
            probs[trial_idx, seq_pos, :] = 0.0
            if pd.isna(raw_resp):
                probs[trial_idx, seq_pos, 2] = 1.0
            else:
                probs[trial_idx, seq_pos, RESPONSE_MAP.get(str(raw_resp).strip(), 2)] = 1.0
    return probs


def get_vlm_responses(
        task: str,
        model_name: str,
        seeds=range(6),
        pad_len: int = 6,
        ignore_learning: bool = True,
        base_path=None,
) -> List[torch.Tensor]:
    """The runs of one VLM on one task, each ``(n_trials, pad_len, 3)`` as ``process_human_responses`` gives.

    Args:
        task: task name, e.g. ``'1back_category'`` (``TaskConfig.task_name``).
        model_name: VLM name, e.g. ``'gemini-3.1-pro-preview'``.
        seeds: run indices.
        ignore_learning: drop the practice trials (session 0).
        base_path: results folder; by default the one of the model's family in data/mturk/vlm.
    """
    base_dir = Path(base_path) if base_path is not None else vlm_results_dir(model_name)
    # the result files are named like the trial lists, with 'pos' and 'obj' for position and identity
    for k, v in {"_position": "_pos", "_identity": "_obj"}.items():
        task = task.replace(k, v)
    prob_tensors = [_load_one_seed(base_dir / f"{model_name}--{seed}" / f"results_{task}_trials.csv",
                                   pad_len, ignore_learning)
                    for seed in seeds]
    n_trials = {p.shape[0] for p in prob_tensors}
    if len(n_trials) > 1:
        raise RuntimeError(f"Trial count differs across runs of {model_name} on {task}: {n_trials}")
    print(f"[VLM] Loaded {len(prob_tensors)} runs of {model_name} on {task}: {prob_tensors[0].shape[0]} trials")
    return prob_tensors
