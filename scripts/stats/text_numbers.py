"""Numbers quoted in the paper text that no figure script prints.

1. Final validation accuracy (the last validation line of each run's out.txt, mean and SD over the 6
   seeds) of the one-hot abstract and the natural ResNet-CLIP models with each sequence core (Sec. 4.1),
   of the LSTM on a frozen random encoder against the pretrained one, and of the dimensionality-matched
   LSTM with noise s = 0.35 (appendix).
2. The human split-half consistency rho(h,h) and the share of it reached by the most human-consistent
   model, the ViT-B/16 DINOv3 LSTM (Discussion).
3. rho(m,h) at matched validation accuracy (about 0.75) of the one-hot abstract LSTM and of the
   dimensionality-matched LSTM without noise, next to the dimensionality-matched LSTM with noise s = 0.35
   and the natural LSTM (appendix, dimensionality matching).
4. MTurk accuracy and rho(m,h) of the random-encoder LSTM, over all trials and per stimulus feature.

The paper reports rho(m,h) clipped at zero after averaging over split halves; both values are printed.

usage: python scripts/stats/text_numbers.py
Reads the training logs in data/checkpoints and the cached responses in data/evaluations (models whose
responses are not cached are evaluated from their checkpoints).
"""
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.experiment_pipeline import run_experiment_pipeline  # noqa: E402
from analysis.factory import AB_CONFIG_DICT, CORES, DIM_MATCH_CONFIG_DICT, NAT_CONFIG_DICT, run_dir  # noqa: E402
from analysis.plot import feature_level_results  # noqa: E402

CONFIGS = {**AB_CONFIG_DICT, **NAT_CONFIG_DICT, **DIM_MATCH_CONFIG_DICT}
VAL_ACC = re.compile(r'^VAL:.*?acc: ([0-9.]+)', re.M)


def final_val_acc(key):
    accs = [float(VAL_ACC.findall((run_dir(CONFIGS[key], s) / 'out.txt').read_text())[-1]) for s in range(6)]
    return np.mean(accs), np.std(accs, ddof=1)


def load(keys):
    return run_experiment_pipeline(
        'text_numbers', ab_model_keys=[k for k in keys if k in AB_CONFIG_DICT],
        nat_model_keys=[k for k in keys if k in NAT_CONFIG_DICT],
        dim_match_model_keys=[k for k in keys if k in DIM_MATCH_CONFIG_DICT], device='cpu')


def rho(cons, key):
    r = cons['models'][key]['mh']['mean']
    return f'{r:+.3f} (reported {max(r, 0.0):.3f})'


def main():
    print('Final validation accuracy, mean +/- SD over seeds')
    for family, keys in [('abstract', [f'AB_{c}_NOMTURK_FINAL' for c in CORES]),
                         ('natural', [f'ResNet_CLIP_{c}_NOMTURK' for c in CORES])]:
        means = []
        for k in keys:
            m, sd = final_val_acc(k)
            means.append(m)
            print(f'  {k:40s} {m:.3f} +/- {sd:.3f}')
        print(f'  {family} cores range from {min(means):.2f} to {max(means):.2f}')
    for k in ['ResNet_CLIP_LSTM_RANDVIS_H8_NOMTURK', 'ResNet_CLIP_LSTM_NOMTURK', 'DIM_MATCH_LSTM_NOMTURK']:
        m, sd = final_val_acc(k)
        print(f'  {k:40s} {m:.3f} +/- {sd:.3f}')

    best, random_encoder = 'ViT_DINOv3_LSTM_NOMTURK', 'ResNet_CLIP_LSTM_RANDVIS_H8_NOMTURK'
    matched = ['AB_LSTM_NOMTURK', 'DIM_MATCH_LSTM_NONOISE_NOMTURK', 'DIM_MATCH_LSTM_NOMTURK', 'ResNet_CLIP_LSTM_NOMTURK']
    data = load(matched + [best, random_encoder])
    cons, acc = data['cons_result']['overall'], data['acc_result']['overall']

    hh = cons['human']['mean']
    best_rho = cons['models'][best]['mh']['mean']
    print(f'\nrho(h,h) {hh:.3f}; {best} rho(m,h) {best_rho:.3f}, {best_rho / hh:.0%} of rho(h,h)')

    print('\nrho(m,h) at matched validation accuracy')
    for k in matched:
        print(f'  {k:40s} {rho(cons, k)}')

    print(f'\n{random_encoder}: MTurk accuracy {acc["models"][random_encoder]["mean"]:.3f}, '
          f'rho(m,h) {rho(cons, random_encoder)}')
    for feature, res in feature_level_results(data).items():
        print(f'  {feature:10s} rho(m,h) {rho(res["cons"]["overall"], random_encoder)}')


if __name__ == '__main__':
    main()
