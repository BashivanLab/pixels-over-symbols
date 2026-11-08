"""Metric and summary logging for training and evaluation.

:class:`MetricLogger` provides smoothed running metrics and a progress-printing
iterator (with timing/memory), while :class:`SummaryLogger` accumulates
per-task, per-feature, per-variation, and per-action accuracy plus stimulus-pair
records, and renders the associated distribution plots.
"""

import datetime
import os
import time
from collections import defaultdict
from typing import List, Dict, Optional, Union

import numpy as np
import pandas as pd
import psutil
import seaborn as sns
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt

from task.dataloader import TASK_IDX_TASK_TYPE
from utils.analysis import pair_dict_to_df
from utils.const import TASK_TYPE_DICT
from utils.helper import get_per_trial_info


def gpu_mem_usage():
    """
    Compute the GPU memory usage for the current device (GB).
    """
    if torch.cuda.is_available():
        mem_usage_bytes = torch.cuda.max_memory_allocated()
    else:
        mem_usage_bytes = 0
    return mem_usage_bytes / 1024 ** 3


def cpu_mem_usage():
    """
    Compute the memory (RAM) usage for the current PROCESS (GB).
    """
    process = psutil.Process(os.getpid())
    # rss: Resident Set Size (non-swapped physical memory a process has used)
    mem_usage_bytes = process.memory_info().rss

    # Optional: If you want to include all your DataLoader worker processes
    for child in process.children(recursive=True):
        try:
            mem_usage_bytes += child.memory_info().rss
        except psutil.NoSuchProcess:
            pass

    usage_gb = mem_usage_bytes / 1024 ** 3
    total_gb = psutil.virtual_memory().total / 1024 ** 3

    return usage_gb, total_gb


class SummaryLogger:
    """
    Logger for tracking trial-level statistics with optional accumulation.

    When accumulate=True: Stores statistics across batches/epochs (for model training)
    When accumulate=False: Only computes and returns statistics (for human data analysis)
    """

    def __init__(
            self,
            hvm_loader_df,
            mapper,
            sort_pair: bool = True,
            accumulate: bool = True,
            save_trials: bool = False,
    ):
        """
        Args:
            hvm_loader_df: DataFrame with stimulus metadata
            mapper: Mapper object for trial info
            accumulate: If True, store statistics across batches/epochs.
                       If False, only compute and return results.
            save_trials: If True (and accumulate=True), save detailed trial info
        """
        self.hvm_loader_df = hvm_loader_df
        self.mapper = mapper
        self.features = ['category', 'identity', 'position']
        self.accumulate = accumulate
        self.save_trials = save_trials and accumulate  # Can only save trials if accumulating
        self.sort_pair = sort_pair
        self.epoch = 0

        # Initialize storage only if accumulating
        if self.accumulate:
            self._init_storage()
            self.reset()

    def _init_storage(self):
        """Initialize storage dictionaries (only used when accumulate=True)."""
        # Per-epoch temporary storage
        self.per_epoch_pair_task_dict = None
        self.per_epoch_pair_feature_dict = None
        self.per_epoch_cat = None
        self.per_epoch_id = None
        self.per_epoch_pos = None
        self.per_epoch_row = None
        self.per_epoch_var = None
        self.per_epoch_task_acc = None
        self.per_epoch_action_acc = None

        # Accumulated across epochs
        self.cat = None
        self.id = None
        self.pos = None
        self.row = None
        self.var = None
        self.task_acc = None
        self.action_acc = None
        self.feature_acc = None
        self.var_acc = None

        # Trial-level storage (optional)
        if self.save_trials:
            self.per_trial_rows = None
            self.pair_task_dict = None
            self.pair_feature_dict = None

    def reset(self):
        """Reset all tracking for a new training run (only when accumulate=True)."""
        if not self.accumulate:
            return

        # Per epoch stimuli count
        self.cat = defaultdict(list)
        self.id = defaultdict(list)
        self.pos = defaultdict(list)
        self.row = defaultdict(list)
        self.var = defaultdict(list)

        # Per epoch accuracy
        self.task_acc = defaultdict(list)
        self.action_acc = defaultdict(list)
        self.feature_acc = defaultdict(list)
        self.var_acc = defaultdict(list)

        self.reset_epoch()

        if self.save_trials:
            self.per_trial_rows = list()
            self.pair_task_dict = {i: dict() for i in range(1, 44)}
            self.pair_feature_dict = {f: dict() for f in self.features}

    def reset_epoch(self):
        """Reset per-epoch tracking dictionaries (only when accumulate=True)."""
        if not self.accumulate:
            return

        self.per_epoch_pair_task_dict = {i: dict() for i in range(1, 44)}
        self.per_epoch_pair_feature_dict = {f: dict() for f in self.features}

        self.per_epoch_cat = defaultdict(int)
        self.per_epoch_id = defaultdict(int)
        self.per_epoch_pos = defaultdict(int)
        self.per_epoch_row = defaultdict(int)
        self.per_epoch_var = defaultdict(int)

        self.per_epoch_task_acc = defaultdict(list)
        self.per_epoch_action_acc = defaultdict(list)

    @torch.no_grad()
    def batch_update(
            self,
            labels: torch.Tensor, outputs: torch.Tensor,
            task_info: List, task_index: torch.Tensor,
            outputs_are_probs: bool = False
    ) -> Dict:
        """
        Process batch and optionally store results.

        Args:
            labels: Ground truth labels (B, T)
            outputs: Model/human outputs (B, T, num_classes) - logits or probabilities
            task_info: Task metadata for each trial in batch
            task_index: Task indices for each trial (B,) or (B, num_tasks)
            outputs_are_probs: If True, outputs are already probabilities

        Returns:
            Dict with 'batch_stats' and 'trial_results' - always returned for inspection
        """
        # Compute all statistics
        results = self._compute_batch_results(
            labels, outputs, task_info, task_index, outputs_are_probs
        )

        # Store if accumulating
        if self.accumulate:
            self._accumulate_results(results)

        return results

    def _compute_batch_results(
            self,
            labels: torch.Tensor, outputs: torch.Tensor,
            task_info: List, task_index: torch.Tensor,
            outputs_are_probs: bool
    ) -> Dict:
        """Compute all batch statistics without storing."""
        # Convert outputs to probabilities if needed
        if outputs_are_probs:
            probs = outputs
        else:
            probs = F.softmax(outputs, dim=-1)

        preds = probs.argmax(dim=-1)

        # Ensure everything is on CPU
        task_index = task_index.cpu()
        labels = labels.cpu()
        preds = preds.cpu()
        probs = probs.cpu()

        B, T = labels.shape

        # Handle one-hot task_index
        if task_index.ndim == 2:
            task_index = task_index.argmax(dim=1) + 1

        # Compute batch-level accuracies
        batch_stats = self._compute_batch_accuracies(labels, preds, task_index)

        # Process individual trials
        trial_results = []
        for batch_i in range(B):
            trial_result = self._compute_trial_result(
                batch_i, task_info, task_index, labels, preds, probs,
                batch_stats['per_trial_acc'][batch_i]
            )
            trial_results.append(trial_result)

        return {
            'batch_stats': batch_stats,
            'trial_results': trial_results
        }

    def _compute_batch_accuracies(
            self,
            labels: torch.Tensor,
            preds: torch.Tensor,
            task_index: torch.Tensor
    ) -> Dict:
        """Compute accuracy statistics for the batch."""
        B, T = labels.shape
        mask = (labels != 2)
        per_frame_correct = (preds == labels).float()

        # Per-trial accuracy
        numer = (per_frame_correct * mask.float()).sum(dim=1).float()
        per_trial_acc = numer / mask.sum(dim=1).float()

        # Flatten for analysis
        labels_flat = labels.contiguous().view(-1)
        preds_flat = preds.contiguous().view(-1)
        mask_flat = mask.contiguous().view(-1)
        task_index_flat = task_index.unsqueeze(1).expand(-1, T).contiguous().view(-1)
        correct = (labels_flat == preds_flat)

        # Task accuracies
        task_accs = {}
        unique_tasks = task_index_flat.unique()
        for task_id in unique_tasks:
            task_mask = (task_index_flat == task_id)
            if task_mask.sum() > 0:
                final_mask = task_mask & mask_flat
                if final_mask.sum() > 0:
                    task_accs[int(task_id.item())] = correct[final_mask].float().mean().item()

        # Action accuracies
        action_accs = {}
        for action in [0, 1]:
            action_mask = (labels_flat == action)
            if action_mask.sum() > 0:
                action_accs[action] = correct[action_mask].float().mean().item()

        return {
            'per_trial_acc': per_trial_acc.cpu().numpy(),
            'task_accs': task_accs,
            'action_accs': action_accs
        }

    def _compute_trial_result(
            self,
            batch_i: int,
            task_info: List, task_index: torch.Tensor,
            labels: torch.Tensor,
            preds: torch.Tensor, probs: torch.Tensor,
            trial_acc: float
    ) -> Dict:
        """Compute all information for a single trial."""
        info_tuple = task_info[batch_i]
        t_id = int(task_index[batch_i].item())

        # Get metadata
        meta_row_lists = get_per_trial_info(
            info_tuple, self.hvm_loader_df, self.mapper
        )

        trial_label_seq = labels[batch_i].cpu().numpy().tolist()
        trial_pred_seq = preds[batch_i].cpu().numpy().tolist()

        # Format confidences
        trial_conf_seq = [
            {action_i: round(c.item(), 5) for action_i, c in enumerate(conf)}
            for conf in probs[batch_i]
        ]

        # Build trial info
        trial_info = {
            "task_id": t_id,
            "trial_label": trial_label_seq,
            "trial_pred": trial_pred_seq,
            "trial_conf": trial_conf_seq,
            "trial_acc": float(trial_acc),
            "unique_key": tuple(meta_row_lists['row_idx'] + [t_id]),
            **meta_row_lists
        }

        # Extract pair information
        mask = (labels[batch_i] != 2)
        pair_info = self._compute_pair_info(
            trial_info, labels[batch_i], preds[batch_i],
            trial_conf_seq, mask, t_id
        )

        # Get stimulus counts
        stim_counts = self._count_stimuli(meta_row_lists)

        return {
            'trial_info': trial_info,
            'pair_info': pair_info,
            'stim_counts': stim_counts
        }

    def _compute_pair_info(
            self,
            trial_info: Dict,
            labels: torch.Tensor,
            preds: torch.Tensor,
            trial_conf_seq: List,
            mask: torch.Tensor,
            task_id: int,
    ) -> Dict:
        """
        Compute stimuli pair information for a trial.

        Returns:
            Dict with 'pairs', 'features', 'action_labels', 'action_preds',
            'action_confs', 'task_id', 'pair_updates'
        """
        task_dataset = TASK_IDX_TASK_TYPE[task_id]
        pairs, features = task_dataset.get_stimuli_pair(trial_info, sort_pair=self.sort_pair)

        action_labels = labels[mask]
        action_preds = preds[mask]
        # Get confidence for the correct action per frame
        action_confs = [
            trial_conf_seq[f_i][labels[f_i].item()]
            for f_i, l in enumerate(labels) if l != 2
        ]

        #  there could be mismatch between label and stim pair order
        #  e.g. ABBA has first item for BB, but stim_pair first item is AA
        action_labels = task_dataset.get_ordered_action(action_labels, task_id)
        action_preds = task_dataset.get_ordered_action(action_preds, task_id)
        action_confs = task_dataset.get_ordered_action(action_confs, task_id)

        # Compute pair updates (what would be added to dicts)
        pair_updates = self._compute_pair_updates(
            pairs, features, action_labels, action_preds, action_confs, task_id
        )

        return {
            'pairs': pairs,
            'features': features,
            'action_labels': action_labels,
            'action_preds': action_preds,
            'action_confs': action_confs,
            'task_id': task_id,
            'pair_updates': pair_updates
        }

    def _compute_pair_updates(
            self, pairs: List, features: List,
            action_labels: torch.Tensor, action_preds: torch.Tensor,
            action_confs: List, task_id: int
    ) -> Dict:
        """
        Compute what would be added to pair dictionaries.
        Returns structured data ready for accumulation.
        """
        task_updates = {}
        feature_updates = defaultdict(dict)

        if task_id in range(37, 41):  # ctxdms tasks
            assert len(pairs) == len(action_labels) == len(action_confs) == len(action_preds) == 1
            assert len(features) == 2

            pair = pairs[-1]
            a_l, a_p, a_c = action_labels[-1], action_preds[-1], action_confs[-1]

            task_updates[pair] = {
                'accuracy': [(a_l == a_p).item()],
                'confidence': [a_c]
            }

            for feature in features:
                feature_updates[feature][pair] = {
                    'accuracy': [(a_l == a_p).item()],
                    'confidence': [a_c]
                }
        else:
            assert len(pairs) == len(features) == len(action_labels) == len(action_preds) == len(action_confs), \
                'All lists should have the same length'

            for pair, feature, a_l, a_p, a_c in zip(
                    pairs, features, action_labels, action_preds, action_confs
            ):
                if pair not in task_updates:
                    task_updates[pair] = {'accuracy': [], 'confidence': []}
                task_updates[pair]['accuracy'].append((a_l == a_p).item())
                task_updates[pair]['confidence'].append(a_c)

                if pair not in feature_updates[feature]:
                    feature_updates[feature][pair] = {'accuracy': [], 'confidence': []}
                feature_updates[feature][pair]['accuracy'].append((a_l == a_p).item())
                feature_updates[feature][pair]['confidence'].append(a_c)

        return {
            'task': task_updates,
            'features': dict(feature_updates)
        }

    def _count_stimuli(self, meta_row_lists: Dict) -> Dict:
        """Count occurrences of each stimulus property."""
        counts = {
            'category': defaultdict(int),
            'identity': defaultdict(int),
            'position': defaultdict(int),
            'row': defaultdict(int),
            'var': defaultdict(int)
        }

        for row in meta_row_lists['row_idx']:
            if row != -1:
                counts['row'][row] += 1
        for cat in meta_row_lists['category']:
            if cat != -1:
                counts['category'][cat] += 1
        for identity in meta_row_lists['identity']:
            if identity != -1:
                counts['identity'][identity] += 1
        for pos in meta_row_lists['position']:
            if pos != -1:
                counts['position'][pos] += 1
        for var in meta_row_lists['var']:
            var = int(var)
            if var != -1:
                counts['var'][var] += 1

        return counts

    @staticmethod
    def results_to_pair_df(results: Dict, split_interdms: bool = True):
        """
        Convert trial results to pair-based dataframes.

        Args:
            results: Dict with 'trial_results' list
            split_interdms: If True, split InterDMS tasks into AA and BB components

        Returns:
            Tuple of (df_pair_tasks, df_pair_features, stim_count_dict)
        """
        # Initialize stimulus counters
        stim_count_dict = {
            key: defaultdict(int)
            for key in ['category', 'identity', 'position', 'row', 'var']
        }

        # Initialize pair dictionaries
        pair_task_dict = SummaryLogger._init_task_dict(split_interdms)
        pair_feature_dict = {
            feature: defaultdict(lambda: {'accuracy': [], 'confidence': []})
            for feature in ['category', 'identity', 'position']
        }

        # Process each trial
        for trial_result in results['trial_results']:
            SummaryLogger._update_stimulus_counts(trial_result, stim_count_dict)
            SummaryLogger._update_pair_dicts(
                trial_result, pair_task_dict, pair_feature_dict, split_interdms
            )

        # Convert to DataFrames
        df_pair_tasks = pair_dict_to_df(pair_task_dict)
        df_pair_features = pair_dict_to_df(pair_feature_dict, feature_dict=True)

        return df_pair_tasks, df_pair_features, stim_count_dict

    @staticmethod
    def _init_task_dict(split_interdms: bool) -> Dict:
        """Initialize task dictionary with appropriate structure for pair tracking."""
        pair_task_dict = {}

        for task_id in range(1, 44):
            if TASK_TYPE_DICT[task_id] == 'InterDMS' and split_interdms:
                # Split InterDMS into AA and BB components
                pair_task_dict[f'{task_id}_AA'] = defaultdict(
                    lambda: {'accuracy': [], 'confidence': []}
                )
                pair_task_dict[f'{task_id}_BB'] = defaultdict(
                    lambda: {'accuracy': [], 'confidence': []}
                )
            else:
                pair_task_dict[task_id] = defaultdict(
                    lambda: {'accuracy': [], 'confidence': []}
                )

        return pair_task_dict

    @staticmethod
    def _update_stimulus_counts(trial_result: Dict, stim_count_dict: Dict):
        """Update stimulus occurrence counts from a single trial."""
        for key, counts_dict in trial_result['stim_counts'].items():
            for stim_id, count in counts_dict.items():
                stim_count_dict[key][stim_id] += count

    @staticmethod
    def _update_pair_dicts(
            trial_result: Dict,
            pair_task_dict: Dict,
            pair_feature_dict: Dict,
            split_interdms: bool
    ):
        """Update pair dictionaries with results from a single trial."""
        pair_info = trial_result['pair_info']
        task_id = pair_info['task_id']
        pair_updates = pair_info['pair_updates']

        # Update task-specific pairs
        if TASK_TYPE_DICT[task_id] == 'InterDMS' and split_interdms:
            SummaryLogger._update_interdms_pairs(task_id, pair_updates['task'], pair_task_dict)
        else:
            SummaryLogger._update_regular_pairs(task_id, pair_updates['task'], pair_task_dict)

        # Update feature pairs (same logic for all tasks)
        for feature, pairs_dict in pair_updates['features'].items():
            SummaryLogger._update_regular_pairs(feature, pairs_dict, pair_feature_dict)

    @staticmethod
    def _update_interdms_pairs(task_id: int, task_pairs: Dict, pair_task_dict: Dict):
        """
        Update InterDMS task pairs by splitting into AA and BB components.

        InterDMS tasks have two stimulus pairs presented in different temporal patterns
        (AABB, ABAB, or ABBA). This splits them for separate tracking.

        Handles two cases:
        1. Two different pairs: each gets assigned to AA or BB
        2. Same pair used twice: split the accuracy/confidence lists between AA and BB
        """
        num_unique_pairs = len(task_pairs)

        if num_unique_pairs == 2:
            pair_items = list(task_pairs.items())

        elif num_unique_pairs == 1:
            # Special case: same pair used for both AA and BB
            pair_key = list(task_pairs.keys())[0]
            updates = task_pairs[pair_key]

            # Verify we have data for both AA and BB (should have 2 accuracy values)
            acc_len = len(updates['accuracy'])
            conf_len = len(updates['confidence'])

            if acc_len != 2 or conf_len != 2:
                raise ValueError(
                    f"InterDMS task {task_id} with single pair should have 2 accuracy/confidence "
                    f"values, got {acc_len} accuracy and {conf_len} confidence: {task_pairs}"
                )

            # Split the data into two separate items
            pair_items = [
                (pair_key, {
                    'accuracy': [updates['accuracy'][0]],
                    'confidence': [updates['confidence'][0]]
                }),
                (pair_key, {
                    'accuracy': [updates['accuracy'][1]],
                    'confidence': [updates['confidence'][1]]
                })
            ]

        else:
            raise ValueError(
                f"InterDMS task {task_id} should have 1 or 2 unique pairs, "
                f"got {num_unique_pairs}: {task_pairs}"
            )

        # First pair goes to AA component, second to BB component
        for suffix, (pair_key, updates) in zip(['AA', 'BB'], pair_items):
            dict_key = f'{task_id}_{suffix}'
            SummaryLogger._extend_pair_data(
                pair_task_dict[dict_key][pair_key],
                updates
            )

    @staticmethod
    def _update_regular_pairs(
            key: Union[int, str],
            pairs_dict: Dict,
            target_dict: Dict
    ):
        """Update regular task or feature pairs without splitting."""
        for pair_key, updates in pairs_dict.items():
            SummaryLogger._extend_pair_data(
                target_dict[key][pair_key],
                updates
            )

    @staticmethod
    def _extend_pair_data(target: Dict, updates: Dict):
        """Extend accuracy and confidence lists for a pair."""
        target['accuracy'].extend(updates['accuracy'])
        target['confidence'].extend(updates['confidence'])

    def _accumulate_results(self, results: Dict):
        """Store computed results (only called when accumulate=True)."""
        batch_stats = results['batch_stats']

        # Update task and action accuracies
        for task_id, acc in batch_stats['task_accs'].items():
            self.per_epoch_task_acc[task_id].append(acc)

        for action_id, acc in batch_stats['action_accs'].items():
            self.per_epoch_action_acc[action_id].append(acc)

        stim_key_mapping = {
            'category': 'per_epoch_cat',
            'identity': 'per_epoch_id',
            'position': 'per_epoch_pos',
            'row': 'per_epoch_row',
            'var': 'per_epoch_var'
        }
        # Process each trial
        for trial_result in results['trial_results']:
            # Update stimulus counts
            for key, counts_dict in trial_result['stim_counts'].items():
                storage_attr = stim_key_mapping[key]
                storage = getattr(self, storage_attr)
                for stim_id, count in counts_dict.items():
                    storage[stim_id] += count

            # Update pair dictionaries
            pair_info = trial_result['pair_info']
            task_id = pair_info['task_id']
            pair_updates = pair_info['pair_updates']

            # Update task pairs
            for pair, updates in pair_updates['task'].items():
                if pair not in self.per_epoch_pair_task_dict[task_id]:
                    self.per_epoch_pair_task_dict[task_id][pair] = {
                        'accuracy': [],
                        'confidence': []
                    }
                self.per_epoch_pair_task_dict[task_id][pair]['accuracy'].extend(updates['accuracy'])
                self.per_epoch_pair_task_dict[task_id][pair]['confidence'].extend(updates['confidence'])

            # Update feature pairs
            for feature, pairs_dict in pair_updates['features'].items():
                for pair, updates in pairs_dict.items():
                    if pair not in self.per_epoch_pair_feature_dict[feature]:
                        self.per_epoch_pair_feature_dict[feature][pair] = {
                            'accuracy': [],
                            'confidence': []
                        }
                    self.per_epoch_pair_feature_dict[feature][pair]['accuracy'].extend(updates['accuracy'])
                    self.per_epoch_pair_feature_dict[feature][pair]['confidence'].extend(updates['confidence'])

            # Save trial info if requested
            if self.save_trials:
                self.per_trial_rows.append(trial_result['trial_info'])

    @torch.no_grad()
    def epoch_update(self):
        """
        Update main dictionaries with per-epoch data and compute summary statistics.
        Only callable when accumulate=True.
        """
        if not self.accumulate:
            raise RuntimeError("epoch_update() can only be called when accumulate=True")

        if self.save_trials:
            self._update_trial_pair_dicts()

        # Compute epoch-level statistics
        df_pair_tasks = pair_dict_to_df(self.per_epoch_pair_task_dict)
        df_pair_features = pair_dict_to_df(self.per_epoch_pair_feature_dict, True)

        # Compute variance accuracy
        self._compute_var_accuracy(df_pair_tasks)

        # Compute feature accuracy
        self._compute_feature_accuracy(df_pair_features)

        # Update stimulus counts
        self._update_epoch_counts()

        # Update task and action accuracies
        self._update_epoch_accuracies()

        # Reset for next epoch
        self.reset_epoch()
        self.epoch += 1

    def _update_trial_pair_dicts(self):
        """Update trial-level pair dictionaries (if saving trials)."""
        # Update task pair dictionaries
        for task_id in range(1, 44):
            for pair, data in self.per_epoch_pair_task_dict[task_id].items():
                if pair not in self.pair_task_dict[task_id]:
                    self.pair_task_dict[task_id][pair] = {
                        'accuracy': [],
                        'confidence': []
                    }
                if data['accuracy']:
                    self.pair_task_dict[task_id][pair]['accuracy'].append(
                        np.mean(data['accuracy'])
                    )
                if data['confidence']:
                    self.pair_task_dict[task_id][pair]['confidence'].append(
                        np.mean(data['confidence'])
                    )

        # Update feature pair dictionaries
        for feature in self.features:
            for pair, data in self.per_epoch_pair_feature_dict[feature].items():
                if pair not in self.pair_feature_dict[feature]:
                    self.pair_feature_dict[feature][pair] = {
                        'accuracy': [],
                        'confidence': []
                    }
                if data['accuracy']:
                    self.pair_feature_dict[feature][pair]['accuracy'].append(
                        np.mean(data['accuracy'])
                    )
                if data['confidence']:
                    self.pair_feature_dict[feature][pair]['confidence'].append(
                        np.mean(data['confidence'])
                    )

    def _compute_var_accuracy(self, df_pair_tasks: pd.DataFrame):
        """Compute accuracy grouped by variation."""
        var_arr = self.hvm_loader_df['var'].to_numpy()
        df_pair_tasks['var_tuple'] = df_pair_tasks['pair_key'].map(
            lambda p: tuple(var_arr[stim_idx] for stim_idx in p)
        )

        # Extract rows where all elements have the same var
        for v in sorted(set(var_arr)):
            v = int(v)
            same_var_df = df_pair_tasks[
                df_pair_tasks['var_tuple'].apply(lambda t: all(x == v for x in t))
            ]
            if len(same_var_df) > 0:
                self.var_acc[v].append(np.concatenate(same_var_df['acc'].values).mean())

    def _compute_feature_accuracy(self, df_pair_features: pd.DataFrame):
        """Compute accuracy grouped by feature."""
        for f in self.features:
            df_f = df_pair_features[df_pair_features['feature'] == f]
            if len(df_f) > 0:
                self.feature_acc[f].append(np.concatenate(df_f['acc'].values).mean())

    def _update_epoch_counts(self):
        """Append this epoch's stimulus counts, so that ``history[epoch]`` is that epoch's count.

        A stimulus not drawn in an epoch gets a 0 for it (this happens with small
        datasets; with the paper's 256k trials every stimulus is drawn every epoch).
        """
        for history, counts in ((self.cat, self.per_epoch_cat), (self.id, self.per_epoch_id),
                                (self.pos, self.per_epoch_pos), (self.row, self.per_epoch_row),
                                (self.var, self.per_epoch_var)):
            for key in set(history) | set(counts):
                history[key] += [0] * (self.epoch - len(history[key]))
                history[key].append(counts.get(key, 0))

    def _update_epoch_accuracies(self):
        """Update epoch-level accuracies."""
        for task_id, acc_list in self.per_epoch_task_acc.items():
            if acc_list:
                self.task_acc[task_id].append(np.mean(acc_list))
        for action_id, acc_list in self.per_epoch_action_acc.items():
            if acc_list:
                self.action_acc[action_id].append(np.mean(acc_list))

    @torch.no_grad()
    def get_summary_stats(self) -> Dict:
        """
        Return summary statistics for the last epoch.
        Only callable when accumulate=True.

        Returns:
            Dict with keys:
                - 'var': {var_id: last_epoch_accuracy or nan}
                - 'task_group': {group_name: mean_accuracy_across_tasks or nan}
                - 'feature': {feature_name: last_epoch_accuracy or nan}
                - 'action': {action_id: last_epoch_accuracy or nan}
        """
        if not self.accumulate:
            raise RuntimeError("get_summary_stats() can only be called when accumulate=True")

        stats = {}

        # Per-var: take last entry
        var_stats = {
            int(v): (history[-1] if history and history[-1] is not None else float('nan'))
            for v, history in self.var_acc.items()
        }
        for v in set(self.hvm_loader_df['var'].to_numpy()):
            v = int(v)
            var_stats.setdefault(v, float('nan'))
        stats['var'] = var_stats

        # Per-feature: last appended
        feat_stats = {
            f: (hist[-1] if hist else float('nan'))
            for f, hist in self.feature_acc.items()
        }
        for f in self.features:
            feat_stats.setdefault(f, float('nan'))
        stats['feature'] = feat_stats

        # Per-action: last appended
        action_stats = {
            a: (hist[-1] if hist else float('nan'))
            for a, hist in self.action_acc.items()
        }
        for a in [0, 1]:
            action_stats.setdefault(a, float('nan'))
        stats['action'] = action_stats

        # Per-task-group: average last per-task accuracies
        group_accs = defaultdict(list)
        for task_id in range(1, 44):
            group = TASK_TYPE_DICT[task_id]
            hist = self.task_acc.get(task_id, [])
            if hist:
                last = hist[-1]
                if last is not None and not (isinstance(last, float) and np.isnan(last)):
                    group_accs[group].append(float(last))

        group_mean = {
            g: (float(np.mean(vals)) if vals else float('nan'))
            for g, vals in group_accs.items()
        }

        # Ensure all groups are present
        unique_groups = set(TASK_TYPE_DICT.values())
        for g in unique_groups:
            group_mean.setdefault(g, float('nan'))

        stats['task_group'] = group_mean

        return stats

    def plot_feature_sampling(
            self,
            feature: str = 'category',
            epoch: Optional[int] = None,
            show: bool = True,
            save_path: Optional[str] = None
    ):
        """
        Plot the sampling distribution for a feature's counts.
        Only callable when accumulate=True.

        Args:
            feature: one of 'category', 'identity', 'position', 'row', 'var'
            epoch: int or None (None -> last epoch). epoch is 0-based.
            show: whether to call plt.show()
            save_path: if provided, save figure to this path

        Returns:
            matplotlib.figure.Figure
        """
        if not self.accumulate:
            raise RuntimeError("plot_feature_sampling() can only be called when accumulate=True")

        mapping = {
            'category': self.cat,
            'identity': self.id,
            'position': self.pos,
            'row': self.row,
            'var': self.var,
        }
        if feature not in mapping:
            raise ValueError(f"feature must be one of {list(mapping.keys())}")

        storage = mapping[feature]
        epoch_idx = -1 if epoch is None else int(epoch)

        # Build counts
        keys = sorted(storage.keys())
        counts = []
        for key in keys:
            history = storage[key]
            cnt = history[epoch_idx] if history else 0
            counts.append(int(cnt))

        # Plot
        fig, ax = plt.subplots(figsize=(8, 4))
        df_plot = pd.DataFrame({feature: keys, 'count': counts})

        sns.barplot(x=feature, y='count', data=df_plot, ax=ax, color='skyblue')
        ax.set_title(f"Sampling Count for {feature} (epoch {epoch_idx})")
        ax.set_ylabel("count")
        ax.set_xlabel(feature)

        if feature == 'row':
            ax.set_xlabel("")
            ax.set_xticks([])  # Often rows are too numerous to show individual ticks
        else:
            ax.set_xlabel(feature)
            # Rotate labels if there are many categories
            if len(keys) > 10:
                plt.xticks(rotation=45, ha='right')

        plt.tight_layout()
        if save_path:
            fig.savefig(save_path, bbox_inches='tight')
        if show:
            plt.show()
        plt.close(fig)
        return fig

    def save_pair_dfs(self, folder: str, prefix: str = "train"):
        """
        Save pair dataframes and trial information to disk.
        Only callable when accumulate=True and save_trials=True.
        """
        if not self.accumulate:
            raise RuntimeError("save_pair_dfs() can only be called when accumulate=True")

        if self.save_trials:
            df_tasks = pair_dict_to_df(self.pair_task_dict)
            task_csv = os.path.join(folder, f"{prefix}_pair_task_df_epoch{self.epoch}.csv")
            task_pkl = os.path.join(folder, f"{prefix}_pair_task_df_epoch{self.epoch}.pkl")
            df_tasks.to_csv(task_csv, index=False)
            df_tasks.to_pickle(task_pkl)

            df_features = pair_dict_to_df(self.pair_feature_dict, True)
            feat_csv = os.path.join(folder, f"{prefix}_pair_feature_df_epoch{self.epoch}.csv")
            feat_pkl = os.path.join(folder, f"{prefix}_pair_feature_df_epoch{self.epoch}.pkl")
            df_features.to_csv(feat_csv, index=False)
            df_features.to_pickle(feat_pkl)

            trials_pth = os.path.join(folder, f"{prefix}_per_trial_rows_epoch{self.epoch}.pth")
            torch.save(self.per_trial_rows, trials_pth)


class MetricLogger:
    """Logger for tracking metrics during training with timing information."""

    def __init__(self, delimiter: str = "\t"):
        self.meters = dict()
        self.delimiter = delimiter
        self.epoch = 0
        self._iter_times = []
        self._data_times = []

    def update(self, **kwargs):
        """Update meters with new values."""
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.cpu().item()
            assert isinstance(v, (float, int)), f"Metric {k} must be float or int, got {type(v)}"
            self.meters[k] = v

    def __str__(self):
        """String representation of current meters."""
        parts = [f"{k}: {v:.4f}" for k, v in sorted(self.meters.items())]
        return self.delimiter.join(parts)

    def _fmt(self, x, fmt: str = "{:.4f}"):
        """Format a value for display."""
        try:
            if x is None:
                return "n/a"
            if isinstance(x, float) and np.isnan(x):
                return "n/a"
            return fmt.format(x)
        except Exception:
            return str(x)

    def _capture_system(self):
        """Capture CPU and GPU memory usage."""
        cpu_mem, cpu_mem_all = cpu_mem_usage()
        self.meters['cpu_mem'] = cpu_mem
        self.meters['gpu_mem'] = gpu_mem_usage()

    def log_every(self, iterable):
        """
        Yield items from iterable while recording timing information.

        Records:
            - data_time: time between iterations
            - iter_time: time spent in user's block

        Usage:
            for batch in metric_logger.log_every(train_loader):
                # process batch
                metric_logger.update(loss=..., acc=...)
        """
        self._iter_times = []
        self._data_times = []

        end = time.time()
        for obj in iterable:
            data_time = time.time() - end
            self._data_times.append(data_time)

            start_iter = time.time()
            yield obj

            iter_time = time.time() - start_iter
            self._iter_times.append(iter_time)
            end = time.time()

    def print_epoch_summary(
            self,
            header: Optional[str] = None,
            print_freq: int = 5,
            summary_logger: Optional[SummaryLogger] = None,
            scheduler=None
    ):
        """
        Print the epoch's timings and metrics, and every ``print_freq`` epochs the
        per-variation, task-group, feature and action accuracies.

        Args:
            header: Optional header string
            print_freq: Print detailed summary every N epochs
            summary_logger: Optional SummaryLogger for detailed stats
            scheduler: Optional learning rate scheduler
        """
        if header is None:
            header = ""
        header = header.ljust(8)

        total_iters = len(self._iter_times)
        avg_iter_time = float(np.mean(self._iter_times)) if total_iters > 0 else 0.0
        avg_data_time = float(np.mean(self._data_times)) if len(self._data_times) > 0 else 0.0
        total_time = sum(self._iter_times) + sum(self._data_times)

        self._capture_system()

        stats = None
        if summary_logger is not None and summary_logger.accumulate:
            stats = summary_logger.get_summary_stats()
        if scheduler is not None:
            self.meters['lr'] = scheduler.get_last_lr()[0]

        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(
            header,
            f"Epoch {self.epoch}",
            f"TOTAL TIME: {total_time_str}  avg_iter: {avg_iter_time:.4f}s, avg_data: {avg_data_time:.4f}s"
        )

        # Metrics line (keep ordering)
        if self.meters:
            order = ['loss', 'acc', 'cpu_mem', 'gpu_mem', 'lr', ]
            parts = []
            for k in order:
                if k in self.meters:
                    v = self.meters[k]
                    if k == 'lr':
                        parts.append(f"{k}: {v:.6f}" if isinstance(v, (int, float)) else f"{k}: {v}")
                    else:
                        parts.append(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")
            for k in sorted(self.meters.keys()):
                if k not in order:
                    v = self.meters[k]
                    parts.append(f"{k}: {v:.4f}" if isinstance(v, (int, float)) else f"{k}: {v}")
            print(header, " | ".join(parts))
        else:
            print(header, "METRICS: (none)")

        # Print SummaryLogger section if available
        if stats is not None and (self.epoch + 1) % print_freq == 0:
            print("-" * 90)
            var = stats.get('var', {})
            if var:
                var_items = ", ".join(
                    f"var{int(k)}:{(v if v is not None else float('nan')):.3f}"
                    for k, v in sorted(var.items())
                )
                print(header, f" VARS: {var_items}")
            tg = stats.get('task_group', {})
            if tg:
                tg_items = " | ".join(
                    f"{g}:{(v if v is not None else float('nan')):.3f}"
                    for g, v in tg.items()
                )
                print(header, f" TASK_GROUPS: {tg_items}")
            f = stats.get('feature', {})
            if f:
                f_items = " | ".join(
                    f"{k}:{(v if v is not None else float('nan')):.3f}"
                    for k, v in f.items()
                )
                print(header, f" FEATURES: {f_items}")
            a = stats.get('action', {})
            if a:
                a_items = " | ".join(
                    f"a{int(k)}:{(v if v is not None else float('nan')):.3f}"
                    for k, v in sorted(a.items())
                )
                print(header, f" ACTIONS: {a_items}")
            print("-" * 90 + '\n')

        self.epoch += 1
