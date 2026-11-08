"""Human responses on the MTurk trials.

A participant file (data/mturk/human/<task_name>/sNN.csv) is the PsychoPy log of one session: a trial
starts at a row with ``TrialIntro.started`` set, the next rows log the following stimuli (``resp.keys``:
'x' = match, 'b' = non-match, empty = no answer), and the row after the last stimulus holds the trial's
``session`` (0 = practice) and correct answers (``act<i>``).
"""

from functools import partial
from pathlib import Path
from typing import List

import pandas as pd
import torch

from mturk.io import mturk_localize
from mturk.task_configs import TaskConfig


def analyze_accuracy(
        behaviour_df: pd.DataFrame,
        trial_df: pd.DataFrame,
        meta_df: pd.DataFrame,
        task_config: TaskConfig,
        ignore_learning: bool = False,
        print_acc: bool = True,
):
    """Score one participant's responses against the trial list.

    A missing answer counts as wrong. Checks that the correct answers logged by PsychoPy are those of
    the trial list, trial by trial.

    Returns:
        ``(summary_df, detail_df)``: one row per trial, and one row per graded response with its
        ``trial_id`` (1-based, after dropping practice trials if ``ignore_learning``), ``resp_index``
        (0-based stimulus index), response, correctness and the graded stimulus with its features.
    """
    n_frames = len(task_config.stim_cols)
    if task_config.trial_start_col not in behaviour_df.columns:
        raise ValueError(f"Column '{task_config.trial_start_col}' not found in behaviour_df")

    meta_lookup = meta_df.set_index('filename').to_dict(orient="index")

    trial_start_indices = behaviour_df.index[
        behaviour_df[task_config.trial_start_col].notna()
    ].tolist()

    responses_data = []
    for i in range(len(trial_start_indices)):
        start_idx = trial_start_indices[i]
        # act_indices are the 1-based positions of the graded stimuli; the response to stimulus k is
        # logged k - 1 rows after the trial start, the correct answers after the last stimulus
        resp_indices = [r_i + start_idx - 1 for r_i in task_config.act_indices]
        act_idx = start_idx + n_frames

        if act_idx >= len(behaviour_df):
            break

        responses = behaviour_df.loc[resp_indices, task_config.resp_col].tolist()
        if task_config.rt_col in behaviour_df.columns:
            rts = behaviour_df.loc[resp_indices, task_config.rt_col].tolist()
        else:
            rts = [None] * len(resp_indices)

        correct_answers = behaviour_df.loc[act_idx, task_config.act_cols].tolist()
        session = int(behaviour_df.loc[act_idx, task_config.session_col])
        responses_data.append({
            "trial_id": i + 1,
            "session": session,
            "start_idx": start_idx,
            "resp_indices": resp_indices,
            "act_idx": act_idx,
            "responses": responses,
            "rts": rts,
            "correct_answers": correct_answers,
        })

    responses_df = pd.DataFrame(responses_data)

    if ignore_learning:
        responses_df = responses_df[responses_df["session"] != 0].reset_index(drop=True)
        trial_df_filtered = trial_df[trial_df["session"] != 0].reset_index(drop=True)
    else:
        trial_df_filtered = trial_df.copy()

    if len(responses_df) != len(trial_df_filtered):
        raise RuntimeError(
            f'trial length do not match between mturk response and trial csv: '
            f'{len(responses_df)} vs {len(trial_df_filtered)}'
        )
    responses_df['trial_id'] = responses_df.index + 1
    for idx, row in responses_df.iterrows():
        trial_csv_answers = trial_df_filtered.loc[idx, task_config.act_cols].tolist()
        if row["correct_answers"] != trial_csv_answers:
            raise RuntimeError(
                f'Trial {idx}: correct answers mismatch: '
                f'{row["correct_answers"]} vs. {trial_csv_answers}'
            )

    trials = []
    details = []

    for idx, row in responses_df.iterrows():
        cond_row = trial_df_filtered.iloc[idx]
        responses = row["responses"]
        rts = row["rts"]
        correct_answers = row["correct_answers"]
        resp_indices = row["resp_indices"]
        session = row["session"]

        trial_correct = []
        response_missing = 0
        for r, c in zip(responses, correct_answers):
            r_stripped = None if pd.isna(r) else str(r).strip()
            c_stripped = None if pd.isna(c) else str(c).strip()

            if r_stripped in (None, ""):
                response_missing += 1

            if r_stripped is None or c_stripped is None or r_stripped == "":
                trial_correct.append(False)
            else:
                trial_correct.append(r_stripped.lower() == c_stripped.lower())

        n_correct = sum(trial_correct)
        n_total = len(trial_correct)
        accuracy = n_correct / n_total if n_total > 0 else float("nan")

        trial_info = {
            "trial_id": row["trial_id"],
            "session": session,
            "n_correct": n_correct,
            "n_total": n_total,
            "accuracy": accuracy,
            "n_missing": response_missing,
        }
        trial_info.update(cond_row[task_config.stim_cols].to_dict())
        trial_info.update(cond_row[task_config.act_cols].to_dict())

        trials.append(trial_info)

        for r_idx, r_row_idx, r, rt, c, corr in zip(
                task_config.act_indices, resp_indices, responses, rts, correct_answers, trial_correct):
            r_stripped = None if pd.isna(r) else str(r).strip()
            is_missing = (r_stripped in (None, ""))

            # the graded stimulus
            stim_col = f"stim{r_idx}"
            stim_path = Path(cond_row.get(stim_col, None))
            stim_path = stim_path.with_name(stim_path.name.lstrip("_"))
            meta_row = meta_lookup[stim_path]
            feature_values = dict()
            for feat in task_config.features:
                feature_values[feat] = meta_row.get(feat, None)

            details_row = {
                "trial_id": row["trial_id"],
                "session": session,
                "resp_index": r_idx - 1,
                "csv_row": r_row_idx,
                "response": r,
                "rt": rt,
                "correct_answer": c,
                "is_correct": corr,
                "is_missing": is_missing,
                "stim_col": stim_col,
                "stim_path": stim_path,
            }
            details_row.update(feature_values)
            details.append(details_row)

    summary_df = pd.DataFrame(trials)
    detail_df = pd.DataFrame(details)

    if print_acc:
        learning_suffix = " (excluding learning trials)" if ignore_learning else ""
        print(f"\nAnalyzed {len(summary_df)} trials{learning_suffix}")
        print(f"Overall (mean per-trial) accuracy: {summary_df['accuracy'].mean():.2%}")
        print(f"Overall (per-frame) accuracy:      {detail_df['is_correct'].mean():.2%}\n")

    return summary_df, detail_df


def get_human_responses(
        task_config: TaskConfig,
        behaviour_dir: Path,
        ignore_learning=True,
        print_acc=False,
):
    """``detail_df`` of :func:`analyze_accuracy` for every participant file of a task, in sorted file order."""
    meta_df = pd.read_csv(task_config.task_meta_csv)
    meta_df['filename'] = meta_df['filename'].apply(partial(mturk_localize, Path('images')))
    trial_df = pd.read_csv(task_config.task_trial_csv)
    human_detail_dfs = list()
    for f in sorted(behaviour_dir.glob('*.csv')):
        _, detail_df = analyze_accuracy(
            behaviour_df=pd.read_csv(f),
            trial_df=trial_df,
            meta_df=meta_df,
            task_config=task_config,
            ignore_learning=ignore_learning,
            print_acc=print_acc
        )
        human_detail_dfs.append(detail_df)
    print(f'Found {len(human_detail_dfs)} participants')
    return human_detail_dfs


def process_human_responses(
        dfs: List[pd.DataFrame],
        trial_len: int,
        pad_len: int = 6
):
    """Participants' responses as ``(n_participants, n_trials, pad_len, 3)`` one-hot over ``[b, x, no response]``.

    The response to stimulus k goes to frame k - 1, the one to the last stimulus to the last frame
    (``pad_len - 1``), as in ``mturk.trials.mturk_trial_to_batches``; the other frames stay at
    "no response".
    """
    response_map = {'b': 0, 'x': 1}
    n_workers = len(dfs)

    trial_ids = sorted(dfs[0]['trial_id'].unique())
    n_trials = len(trial_ids)

    col_idx_map = {i: i for i in range(trial_len - 1)}
    col_idx_map[trial_len - 1] = pad_len - 1

    per_subject_out = torch.zeros((n_workers, n_trials, pad_len, 3))
    per_subject_out[:, :, :, 2] = 1.0

    for worker_idx, df in enumerate(dfs):
        trial_groups = df.groupby('trial_id')

        for trial_idx, t_id in enumerate(trial_ids):
            if t_id not in trial_groups.groups:
                continue

            trial_df = trial_groups.get_group(t_id)

            for resp_index in range(trial_len):
                target_frame_idx = col_idx_map[resp_index]
                frame_df = trial_df[trial_df['resp_index'] == resp_index]

                if not frame_df.empty:
                    response = frame_df.iloc[0]['response']
                    per_subject_out[worker_idx, trial_idx, target_frame_idx, :] = 0.0

                    if pd.isna(response):
                        per_subject_out[worker_idx, trial_idx, target_frame_idx, 2] = 1.0
                    else:
                        cls_idx = response_map.get(response, 2)
                        per_subject_out[worker_idx, trial_idx, target_frame_idx, cls_idx] = 1.0

    return per_subject_out
