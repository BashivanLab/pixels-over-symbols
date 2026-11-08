"""Helpers of the behavioral metrics.

Fisher-z averaging of correlations, split halves, and the conversion of ``eval_mturk``'s per-task pair
dataframes into the response vectors the metrics take. A response vector concatenates the tasks in
order of their original id, with each InterDMS task split into its A and B comparisons, and orders
the pairs of a task by ``pair_key``.
"""

from typing import Dict, List, Union

import numpy as np
import pandas as pd
import torch

from utils.const import TASK_METADATA

# the 12 tasks run on MTurk (task ids as in utils.const.TASK_NAME_TASK_INDEX)
HUMAN_TASK_IDS = [1, 2, 3, 19, 23, 27, 28, 32, 36, 37, 40, 41]

EPS = 1e-6


def fisher_z_transform(r: float) -> float:
    """Fisher z-transform, with r clipped to (-1, 1)."""
    r = np.clip(r, -1 + EPS, 1 - EPS)
    return 0.5 * np.log((1 + r) / (1 - r))


def fisher_z_inverse(z: float) -> float:
    """Inverse Fisher z-transform."""
    return np.tanh(z)


def fisher_z_mean(correlations: List[float]) -> float:
    """Average correlations using Fisher z-transform."""
    z_values = [fisher_z_transform(r) for r in correlations]
    mean_z = np.mean(z_values)
    return fisher_z_inverse(mean_z)


def fisher_mean_matrix(matrix_3d, axis=-1, eps=1e-5):
    """Average an array of correlations along `axis` in Fisher z (clipped so that r = 1 stays finite)."""
    clipped_matrix = np.clip(matrix_3d, -1.0 + eps, 1.0 - eps)
    z_mean = np.arctanh(clipped_matrix).mean(axis=axis)
    return np.tanh(z_mean)


def _process_task_dataframe(
        df: pd.DataFrame,
        task_id: int,
        metric: str,
        split_interdms: bool,
        cast_to_int: bool
):
    """The `metric` vector(s) of one task's pair dataframe: [A comparisons, B comparisons] for a split
    InterDMS task, else [all pairs]."""
    vectors = []
    if TASK_METADATA[task_id]['type'] == 'InterDMS' and split_interdms:
        aa_df = df[df['task_id'] == f'{task_id}_AA']
        bb_df = df[df['task_id'] == f'{task_id}_BB']

        for sub_df in [aa_df, bb_df]:
            vectors.append(_extract_metric_vector(sub_df, metric, cast_to_int))
    else:
        vectors.append(_extract_metric_vector(df, metric, cast_to_int))

    return vectors


def _extract_metric_vector(df: pd.DataFrame, metric: str, cast_to_int: bool = False):
    """The `metric` column of a pair dataframe, ordered by pair_key."""
    df_sorted = df.sort_values('pair_key')

    vector = []
    for _, row in df_sorted.iterrows():
        val = row[metric]
        # values may be wrapped in a one-element list or array
        if isinstance(val, (list, np.ndarray)):
            extracted = val[0] if len(val) > 0 else (0 if cast_to_int else np.nan)
        else:
            extracted = val

        if cast_to_int:
            vector.append(int(extracted))
        else:
            vector.append(extracted)

    return vector


def convert_eval_mturk_to_consistency_format(
        task_pair_dfs: Dict[int, Dict[str, List[pd.DataFrame]]],
        model_type: str = 'nat_model',
        metric: str = 'confs',
        aggregate_tasks: bool = True,
        split_interdms: bool = True,
) -> Union[Dict[str, np.ndarray], Dict[int, Dict[str, np.ndarray]]]:
    """Response vectors of one responder from ``eval_mturk``'s ``task_pair_dfs``.

    Args:
        task_pair_dfs: second output of ``eval_mturk``.
        model_type: the responder (model key or VLM name).
        metric: 'confs' (confidence in the correct answer) or 'acc' (correctness).
        aggregate_tasks: concatenate the tasks into one vector per seed.
        split_interdms: split each InterDMS task into its A and B comparisons.

    Returns:
        ``{'<model_type>_<seed>': (n_trials,)}``, or ``{task_id: {...}}`` without ``aggregate_tasks``.
    """
    first_task_id = list(task_pair_dfs.keys())[0]
    n_instances = len(task_pair_dfs[first_task_id][model_type])

    if aggregate_tasks:
        model_outputs = {}

        for instance_idx in range(n_instances):
            seed_id = f'{model_type}_{instance_idx}'
            all_vectors = []

            for task_id in sorted(task_pair_dfs.keys()):
                df = task_pair_dfs[task_id][model_type][instance_idx]

                vectors = _process_task_dataframe(
                    df, task_id, metric, split_interdms, cast_to_int=False
                )
                for v in vectors:
                    all_vectors.append(np.array(v))
            model_outputs[seed_id] = np.concatenate(all_vectors)

        return model_outputs
    else:
        per_task_outputs = {}

        for task_id in sorted(task_pair_dfs.keys()):
            task_outputs = {}

            for instance_idx in range(n_instances):
                seed_id = f'{model_type}_{instance_idx}'
                df = task_pair_dfs[task_id][model_type][instance_idx]

                vectors = _process_task_dataframe(
                    df, task_id, metric, split_interdms, cast_to_int=False
                )

                task_outputs[seed_id] = np.concatenate([np.array(v) for v in vectors])

            per_task_outputs[task_id] = task_outputs

        return per_task_outputs


def convert_human_responses_to_matrix(
        task_pair_dfs: Dict[int, Dict[str, List[pd.DataFrame]]],
        aggregate_tasks: bool = True,
        split_interdms: bool = True
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """Participants' correctness from ``eval_mturk``'s ``task_pair_dfs``.

    Returns:
        ``(n_participants, n_trials)`` 0/1 matrix, or ``{task_id: matrix}`` without ``aggregate_tasks``.
    """
    first_task_id = list(task_pair_dfs.keys())[0]
    n_subjects = len(task_pair_dfs[first_task_id]['human'])

    if aggregate_tasks:
        all_subject_responses = []

        for subject_idx in range(n_subjects):
            subject_vector_parts = []

            for task_id in sorted(task_pair_dfs.keys()):
                df = task_pair_dfs[task_id]['human'][subject_idx]

                vectors = _process_task_dataframe(
                    df, task_id, metric='acc', split_interdms=split_interdms, cast_to_int=True
                )

                for v in vectors:
                    subject_vector_parts.extend(v)

            all_subject_responses.append(subject_vector_parts)

        return np.array(all_subject_responses)

    else:
        per_task_matrices = {}

        for task_id in sorted(task_pair_dfs.keys()):
            task_responses = []

            for subject_idx in range(n_subjects):
                df = task_pair_dfs[task_id]['human'][subject_idx]

                vectors = _process_task_dataframe(
                    df, task_id, metric='acc', split_interdms=split_interdms, cast_to_int=True
                )

                full_task_vector = []
                for v in vectors:
                    full_task_vector.extend(v)

                task_responses.append(full_task_vector)

            per_task_matrices[task_id] = np.array(task_responses)

        return per_task_matrices


def convert_task_features_to_matrix(
        task_features: Dict[int, Dict[str, List[List[torch.Tensor]]]],
        model_type: str = 'nat_model',
        has_visual_features: bool = True
) -> Union[np.ndarray, Dict[int, np.ndarray]]:
    """Hidden states of one model from ``eval_mturk(get_act=True)``'s ``task_features``.

    Unlike the response vectors these keep all 6 frames of every trial.

    Returns:
        ``(seq, pre)``: ``{seed: {task_id: (T, n_trials, hidden)}}`` for the output of the sequence core
        and for its input.
    """

    first_task_id = list(task_features.keys())[0]
    n_instances = len(task_features[first_task_id][model_type])

    model_seq_hidden = dict()
    model_pre_hidden = dict()
    for instance_idx in range(n_instances):
        task_seq_vectors = dict()
        task_pre_vectors = dict()
        for task_id in sorted(task_features.keys()):
            feature_tuples = task_features[task_id][model_type][instance_idx]
            if not has_visual_features:
                pre_features, hidden_states = feature_tuples[0], feature_tuples[1]
            else:
                pre_features, hidden_states = feature_tuples[1], feature_tuples[2]

            pre_features = np.stack(pre_features)
            hidden_states = np.stack(hidden_states)
            task_seq_vectors[task_id] = hidden_states
            task_pre_vectors[task_id] = pre_features
        model_seq_hidden[instance_idx] = task_seq_vectors
        model_pre_hidden[instance_idx] = task_pre_vectors
    return model_seq_hidden, model_pre_hidden


def generate_subject_splits(
        n_subjects: int,
        n_splits: int = 10,
        random_state: int = 42
):
    """`n_splits` random halves (seeded) of `n_subjects` participants or model seeds."""
    rng = np.random.RandomState(random_state)
    splits = []

    for _ in range(n_splits):
        idx = rng.permutation(n_subjects)
        half = n_subjects // 2
        splits.append((idx[:half], idx[half:2 * half]))

    return splits


def get_feature_indices(
        feature_pair_dfs,
        task_pair_dfs,
        task_configs,
        task_metadata,
        new_mapping,
        n_pairs=100,
):
    """Indices of the trials comparing each feature (category, identity, position) in the response vectors.

    A context-DMS trial counts toward the feature compared in its decision.

    Returns:
        ``(indices, task_indices, local_indices)``: ``{feature: indices}``, ``{feature: {new task id:
        indices}}`` and the same with indices within the task.
    """
    config_feature_map = {
        'pos_1b': 'position',
        'cat_1b': 'category',
        'obj': 'identity',
    }

    global_indices = {f: [] for f in config_feature_map.values()}
    task_global = {f: {tid: [] for tid in task_metadata} for f in config_feature_map.values()}
    task_local = {f: {tid: [] for tid in task_metadata} for f in config_feature_map.values()}

    for i, (t_id, t_info) in enumerate(task_metadata.items()):
        task_offset = i * n_pairs
        old_t_id = new_mapping[t_id]
        t_type = t_info['type']

        full_task_df = task_pair_dfs[old_t_id]['human'][0]
        feature_df = feature_pair_dfs[old_t_id]['human'][0]

        suffix = None
        if t_type == 'InterDMS':
            suffix = t_info['name'].split('_')[-1]
            assert suffix in ['AA', 'BB']
            target_name = f'{old_t_id}_{suffix}'
        else:
            target_name = str(old_t_id)

        if 'task_id' in full_task_df.columns and full_task_df['task_id'].nunique() > 1:
            # an InterDMS task holds both comparisons
            current_task_df = full_task_df[full_task_df['task_id'] == target_name]
            current_task_df = current_task_df.sort_values('pair_key').reset_index(drop=True)
        else:
            current_task_df = full_task_df.sort_values('pair_key').reset_index(drop=True)

        assert len(current_task_df) == n_pairs, \
            f"Length mismatch for {target_name} (ID {t_id}). Expected {n_pairs}, found {len(current_task_df)}."

        config_feats = [config_feature_map[f] for f in task_configs[old_t_id].features]

        if t_type == 'CtxDMS':
            # the first feature is the context; the decision compares one of the other two
            ctx_feature_name = config_feats[0]
            expected_test_feats = set(config_feats[1:])
            valid_rows = feature_df[feature_df['feature'] != ctx_feature_name]
            assert valid_rows['pair_key'].is_unique, f"Duplicate pair_keys found in CtxDMS {target_name}"
            actual_test_feats = set(valid_rows['feature'].unique())
            assert actual_test_feats == expected_test_feats, \
                f"CtxDMS feature mismatch. Expected {expected_test_feats}, found {actual_test_feats}"

            pair_to_feature = dict(zip(valid_rows['pair_key'], valid_rows['feature']))
            for local_idx, row in current_task_df.iterrows():
                feature = pair_to_feature.get(row['pair_key'])
                global_idx = task_offset + local_idx
                global_indices[feature].append(global_idx)
                task_global[feature][t_id].append(global_idx)
                task_local[feature][t_id].append(local_idx)
        elif t_type == 'InterDMS':
            expected_test_feats = set(config_feats)
            actual_test_feats = set(feature_df['feature'].unique())
            assert actual_test_feats == expected_test_feats, \
                f"InterDMS feature mismatch. Expected {expected_test_feats}, found {actual_test_feats}"

            if suffix == 'AA':
                feature = config_feats[0]
            elif suffix == 'BB':
                feature = config_feats[1]
            else:
                raise RuntimeError('InterDMS config should be AA or BB')

            for local_idx in range(n_pairs):
                global_idx = task_offset + local_idx
                global_indices[feature].append(global_idx)
                task_global[feature][t_id].append(global_idx)
                task_local[feature][t_id].append(local_idx)
        else:
            actual_feats = set(feature_df['feature'].unique())
            assert len(actual_feats) == 1 and list(actual_feats)[0] == config_feats[0], \
                f"Feature mismatch in {t_id}. Expected {config_feats[0]}, found {actual_feats}"
            feature = config_feats[0]
            for local_idx in range(n_pairs):
                global_idx = task_offset + local_idx
                global_indices[feature].append(global_idx)
                task_global[feature][t_id].append(global_idx)
                task_local[feature][t_id].append(local_idx)
    return {k: np.array(v) for k, v in global_indices.items()}, task_global, task_local


def summarize_std(data, is_correlation=True):
    """Mean and std over splits; for correlations both in Fisher z, with the mean transformed back."""
    x = np.array(data)
    if is_correlation:
        z_values = [fisher_z_transform(r) for r in x]
        return {
            'mean': fisher_z_inverse(np.mean(z_values)),
            'std': np.std(z_values),
            'n_splits': len(x)
        }
    return {
        'mean': np.mean(x),
        'std': np.std(x),
        'n_splits': len(x)
    }
