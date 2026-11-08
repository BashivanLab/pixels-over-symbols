"""Conversion of per-pair accuracy/confidence records into DataFrames.

Used by :class:`utils.logger.SummaryLogger` during training and evaluation.
"""

import numpy as np
import pandas as pd


def pair_dict_to_df(
        pair_dict,
        feature_dict: bool = False,
):
    """
    Converts pair_task_dict (task_id -> {pair: {'accuracy':[...], 'confidence':[...]} })
    into a DataFrame with columns:
      - task_id/feature
      - pair_key (tuple)
      - mean_acc (float)
      - mean_conf (float)
      - n_acc (int)  # number of accuracy observations
      - n_conf (int) # number of confidence observations
    """
    rows = []
    if feature_dict:
        for feature, pdict in pair_dict.items():
            for pair, stats in pdict.items():
                accs = np.array(stats.get("accuracy", []), dtype=float)
                confs = np.array(stats.get("confidence", []), dtype=float)
                rows.append({
                    "feature": feature,
                    "pair_key": pair,
                    "mean_acc": float(np.nan) if accs.size == 0 else float(accs.mean()),
                    "mean_conf": float(np.nan) if confs.size == 0 else float(confs.mean()),
                    'acc': accs,
                    'confs': confs,
                    "n_acc": accs.size,
                    "n_conf": confs.size
                })
        df = pd.DataFrame(rows)
    else:
        for task_id, pdict in pair_dict.items():
            for pair, stats in pdict.items():
                accs = np.array(stats.get("accuracy", []), dtype=float)
                confs = np.array(stats.get("confidence", []), dtype=float)
                rows.append({
                    "task_id": task_id,
                    "pair_key": pair,
                    "mean_acc": float(np.nan) if accs.size == 0 else float(accs.mean()),
                    "mean_conf": float(np.nan) if confs.size == 0 else float(confs.mean()),
                    'acc': accs,
                    'confs': confs,
                    "n_acc": accs.size,
                    "n_conf": confs.size
                })
        df = pd.DataFrame(rows)
    return df
