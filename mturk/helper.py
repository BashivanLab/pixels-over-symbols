"""Stimulus pairs of the MTurk trials.

All trial lists compare the same 100 ordered stimulus pairs: 20 chains of 5 pairs in which each pair
starts with the stimulus that ended the previous one (A-B, B-C, ...), so that a chain is one 1-back
trial. :func:`select_balanced_chained_pair` draws such a set with match rates near 0.5 for category,
object and position.
"""

import itertools

import numpy as np
import pandas as pd


def make_ordered_pair_df_with_features(stim_df: pd.DataFrame) -> pd.DataFrame:
    """All ordered pairs (i, j), i != j, of the rows of `stim_df`, with their match features.

    Output columns:
        stim1, stim2 : row indices of the stimuli in stim_df
        same_category, same_obj, same_pos : 0/1 features
        stim{1,2}_category, stim{1,2}_obj, stim{1,2}_pos : the stimuli's features
        stim1_fp, stim2_fp : the stimuli's ``re_filename``
    """
    idx = stim_df.index.tolist()
    pairs = itertools.permutations(idx, 2)

    rows = []
    for i, j in pairs:
        s1 = stim_df.loc[i]
        s2 = stim_df.loc[j]

        same_category = int(s1["cat_1b"] == s2["cat_1b"])
        same_obj = int(s1["id_1b"] == s2["id_1b"] and same_category)
        same_pos = int(s1["pos_1b"] == s2["pos_1b"])

        rows.append(
            {
                "stim1": i,
                "stim2": j,
                "same_category": same_category,
                "same_obj": same_obj,
                "same_pos": same_pos,
                "stim1_category": s1["cat_1b"],
                "stim2_category": s2["cat_1b"],
                "stim1_obj": s1["id_1b"],
                "stim2_obj": s2["id_1b"],
                "stim1_pos": s1["pos_1b"],
                "stim2_pos": s2["pos_1b"],
                "stim1_fp": s1["re_filename"],
                "stim2_fp": s2["re_filename"],
            }
        )

    pair_df = pd.DataFrame(rows)
    return pair_df


def make_chained_pair(
        pair_df: pd.DataFrame,
        n_trials: int,
        n_chains: int,
        feature_cols=("same_category", "same_obj", "same_pos"),
):
    """One random draw of `n_trials` chains of `n_chains` pairs, steered toward match rates of 0.5.

    A chain starts at a random pair; each next pair starts with the previous pair's second stimulus and
    is drawn with probability increasing with how close it brings the running match rates of
    `feature_cols` to 0.5. No two pairs share the same two stimuli.

    Returns:
        trials: list[list[int]] of pair_df row indices, or None if a chain cannot be continued.
    """
    seen_unordered = set()
    trials = []

    # running feature counts over all chains drawn so far
    feature_cols = list(feature_cols)
    feature_counts = {f: 0 for f in feature_cols}

    for t in range(n_trials):
        trial_idxs = []

        for k in range(n_chains):
            if k == 0:
                candidates = pair_df[~pair_df.apply(
                    lambda r: (min(r["stim1"], r["stim2"]),
                               max(r["stim1"], r["stim2"])) in seen_unordered,
                    axis=1
                )]
                row_pick = candidates.sample(n=1)
                idx_pick = row_pick.index[0]
            else:
                prev_idx = trial_idxs[-1]
                s_prev2 = int(pair_df.loc[prev_idx, "stim2"])
                candidates = pair_df[pair_df["stim1"] == s_prev2].copy()
                candidates = candidates[~candidates.apply(
                    lambda r: (min(r["stim1"], r["stim2"]),
                               max(r["stim1"], r["stim2"])) in seen_unordered,
                    axis=1
                )]

                if len(candidates) == 0:
                    return None

                # score every candidate by how close the match rates would be to 0.5 after taking it
                n_current = len(trials) * n_chains + len(trial_idxs)
                feat_matrix = candidates[feature_cols].to_numpy()
                cur_ratios = np.array([feature_counts[f] / max(1, n_current) for f in feature_cols])
                new_ratios = (feat_matrix + cur_ratios * n_current) / (n_current + 1)
                scores = -np.abs(new_ratios - 0.5).sum(axis=1)

                idx_pick = np.random.choice(
                    candidates.index,
                    p=(scores - scores.min() + 1e-6) / np.sum(scores - scores.min() + 1e-6)
                )

            s1 = int(pair_df.loc[idx_pick, "stim1"])
            s2 = int(pair_df.loc[idx_pick, "stim2"])
            seen_unordered.add((min(s1, s2), max(s1, s2)))
            trial_idxs.append(int(idx_pick))

            for f in feature_cols:
                feature_counts[f] += pair_df.loc[idx_pick, f]

        trials.append(trial_idxs)

    return trials


def select_balanced_chained_pair(
        pair_df,
        n_trials=20,
        n_chains=5,
        n_tries_limit=10000,
):
    """The best of up to `n_tries_limit` draws of :func:`make_chained_pair`: match rates closest to 0.5.

    Stops early at an exact 0.5 for all three features.

    Returns:
        ``(trial_df, trials)``: the pairs of the chosen draw, chain by chain, and its chains as lists
        of pair_df row indices.
    """
    best_trial_df, best_trials = None, None
    best_mean = np.array([float('inf'), float('inf'), float('inf')])
    for n_tries in range(n_tries_limit):
        if n_tries % 100 == 0:
            print(f'Try {n_tries}...')
        trials = make_chained_pair(pair_df, n_trials, n_chains)
        if trials is None:
            continue
        flat_trials = [idx for trial in trials for idx in trial]
        trial_df = pair_df.loc[flat_trials].reset_index(drop=True)
        cur_mean = np.array([
            trial_df['same_category'].mean(),
            trial_df['same_obj'].mean(),
            trial_df['same_pos'].mean(),
        ])
        if np.linalg.norm(cur_mean - 0.5) < np.linalg.norm(best_mean - 0.5):
            best_mean = cur_mean
            best_trial_df, best_trials = trial_df, trials
            print(f'New best mean: {best_mean} at try {n_tries}')
            if np.allclose(best_mean, 0.5):
                break
    return best_trial_df, best_trials


def get_task_info_tuples(stim_df, stim_fp):
    """Row index in `stim_df` and (category, identity, position, view angles, size) of a stimulus.

    `stim_fp` is the stimulus's ``local_fp`` (``mturk.io.localize_mturk_csv_filename``).
    """
    stim_row = stim_df[stim_df['local_fp'] == stim_fp]
    assert len(stim_row) == 1, f'{stim_fp} not found or multiple found in stim_df'
    row_idx = stim_row.index[0]
    view_cols = ["rxy", "rxz", "ryz"]
    size_col = "s"
    info_tuple = (
        stim_df.loc[row_idx, 'cat_1b'],
        stim_df.loc[row_idx, 'id_1b'],
        stim_df.loc[row_idx, 'pos_1b'],
        stim_df.loc[row_idx, view_cols].to_list(),
        stim_df.loc[row_idx, size_col],
    )
    return row_idx, info_tuple
