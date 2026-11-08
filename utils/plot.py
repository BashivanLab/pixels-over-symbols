"""Training-curve plots written into each run folder.

:func:`plot_training_stats` draws the per-epoch accuracy breakdowns (per task
group, feature, variation level and action) and saves the average task accuracy
curves as ``{train,val}_avg_acc_rep{r}.npy``.
"""

import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from utils.const import TASK_TYPE_DICT


def _pad_and_stack(lists):
    """Pad lists (may have different lengths) with np.nan and stack to ndarray (n_items, max_len)."""
    if not lists:
        return np.zeros((0, 0))
    L = max(len(x) for x in lists)
    arr = np.array([np.pad(np.asarray(x, dtype=float), (0, L - len(x)), constant_values=np.nan) for x in lists])
    return arr  # shape (n_items, L)


def plot_training_stats(
        stats,
        folder,
        rep_index,
):
    """
    Plot summary-level series produced by SummaryLogger and stored in `stats`.

    Expects `stats` to contain (for train/val prefixes):
      - "{train|val}_task_acc": dict task_id -> [epoch_acc,...]
      - "{train|val}_action_acc": dict action -> [epoch_acc,...]
      - "{train|val}_feature_acc": dict feature -> [epoch_acc,...]
      - "{train|val}_var_acc": dict var -> [epoch_acc,...]

    Uses TASK_TYPE_DICT (task_id -> group name) from the global scope to make group plots.
    """
    os.makedirs(folder, exist_ok=True)

    # TASK GROUP PLOT (uses TASK_TYPE_DICT to map task_id -> group)
    def _plot_task_groups(prefix, fname):
        task_map = stats[f"{prefix}_task_acc"]
        if not task_map:
            return
        # group -> list of per-task series
        group_series = defaultdict(list)
        for task_id, series in task_map.items():
            group = TASK_TYPE_DICT.get(int(task_id), "Unknown")
            if series:
                group_series[group].append(series)

        if not group_series:
            return

        plt.figure(figsize=(10, 6))
        pal = sns.color_palette("tab10", n_colors=max(1, len(group_series)))
        for (grp, lists), c in zip(sorted(group_series.items()), pal):
            arr = _pad_and_stack(lists)  # (n_tasks_in_group, epochs)
            mean = np.nanmean(arr, axis=0)
            plt.plot(mean, label=grp)
        plt.title(f"{prefix.upper()} - Mean Task Accuracy by Task Group")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, fname))
        plt.close()

    # FEATURE-SERIES PLOT (directly from feature_acc summary)
    def _plot_feature_summary(prefix, fname):
        feat_map = stats[f"{prefix}_feature_acc"]
        if not feat_map:
            return
        plt.figure(figsize=(10, 6))
        pal = sns.color_palette("tab10", n_colors=max(1, len(feat_map)))
        for (feat, series), c in zip(sorted(feat_map.items()), pal):
            if not series:
                continue
            plt.plot(np.asarray(series, dtype=float), label=str(feat))
        plt.title(f"{prefix.upper()} - Feature Accuracy (SummaryLogger)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, fname))
        plt.close()

    # VAR-SERIES PLOT
    def _plot_var_summary(prefix, fname):
        var_map = stats[f"{prefix}_var_acc"]
        if not var_map:
            return
        plt.figure(figsize=(8, 5))
        pal = sns.color_palette("tab10", n_colors=max(1, len(var_map)))
        for (v, series), c in zip(sorted(var_map.items()), pal):
            if not series:
                continue
            plt.plot(np.asarray(series, dtype=float), label=f"var {v}")
        plt.title(f"{prefix.upper()} - Var Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, fname))
        plt.close()

    # ACTION-SERIES PLOT
    def _plot_action_summary(prefix, fname):
        a_map = stats[f"{prefix}_action_acc"]
        if not a_map:
            return
        plt.figure(figsize=(8, 5))
        pal = sns.color_palette("tab10", n_colors=max(1, len(a_map)))
        for (a, series), c in zip(sorted(a_map.items()), pal):
            if not series:
                continue
            plt.plot(np.asarray(series, dtype=float), label=f"action {a}")
        plt.title(f"{prefix.upper()} - Action Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(folder, fname))
        plt.close()

    # AVERAGE TASK ACCURACY (global average across tasks)
    def _save_avg_task(prefix, fname_npy, fname_png):
        task_map = stats[f"{prefix}_task_acc"]
        if not task_map:
            return
        lists = [s for s in task_map.values() if len(s) > 0]
        if not lists:
            return
        arr = _pad_and_stack(lists)  # (n_tasks, epochs)
        avg = np.nanmean(arr, axis=0)
        np.save(os.path.join(folder, fname_npy), avg)
        plt.figure(figsize=(8, 4))
        plt.plot(avg)
        plt.title(f"{prefix.upper()} - Average Task Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.tight_layout()
        plt.savefig(os.path.join(folder, fname_png))
        plt.close()

    # Do the plots for TRAIN and VAL
    _plot_task_groups("train", f"group_train_rep{rep_index}.png")
    _plot_task_groups("val", f"group_val_rep{rep_index}.png")

    _plot_feature_summary("train", f"feature_train_summary_rep{rep_index}.png")
    _plot_feature_summary("val", f"feature_val_summary_rep{rep_index}.png")

    _plot_var_summary("train", f"var_train_rep{rep_index}.png")
    _plot_var_summary("val", f"var_val_rep{rep_index}.png")

    _plot_action_summary("train", f"action_train_rep{rep_index}.png")
    _plot_action_summary("val", f"action_val_rep{rep_index}.png")

    _save_avg_task("train", f"train_avg_acc_rep{rep_index}.npy", f"train_avg_acc_rep{rep_index}.png")
    _save_avg_task("val", f"val_avg_acc_rep{rep_index}.npy", f"val_avg_acc_rep{rep_index}.png")
    return
