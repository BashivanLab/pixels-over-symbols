"""Build the 12 MTurk trial lists from a stimulus subset, or check existing ones.

usage: python scripts/mturk/build_trials.py OUT_DIR [--subset CSV] [--seed SEED] [--tries N]
       python scripts/mturk/build_trials.py --check [TRIAL_DIR]

Every trial list compares the same 100 ordered stimulus pairs: 20 chains of 5 pairs, each pair starting
with the stimulus that ended the previous one, drawn with match rates near 0.5 for category, object and
position (mturk.helper.select_balanced_chained_pair: the best of N random draws, about 1.5 s each). From
that set:
    1-back      each chain is one trial of 6 stimuli, in the order drawn
    DMS         each pair is one trial, in shuffled order
    InterDMS    each trial interleaves two pairs (A and B) from two independent shuffles of the set, as
                A1 B1 A2 B2 (ABAB) or A1 B1 B2 A2 (ABBA); every pair occurs once as A and once as B
    context DMS each pair, in shuffled order, follows a context stimulus drawn from the subset that
                matches its first stimulus in the context feature with probability 0.5; the answer
                compares the pair in the second feature if the context matches, else in the third
Trials go to sessions 1-5, 20 pairs per session. A second pair set, drawn the same way, gives the practice
trials (session 0): the first 2 (1-back), 3 (DMS), 5 (InterDMS) or 8 (context DMS) trials of the same list
built from it. The lists are written to OUT_DIR under the names of data/mturk/trials. Stimuli are named
images/<Variation folder>/<file name without its leading underscore>, as exported by build_stimuli.py.

The released lists have this structure, but their pair sets, shuffles and context stimuli came from an
unrecorded random state, and the practice trials of 1back_category and 1back_obj do not follow the rule
above (those of 1back_obj repeat its first two main trials), so a rebuild gives other trials. --check
verifies for a folder of lists (by default data/mturk/trials) that every answer agrees with the stimulus
metadata, that the main trials of every list compare the same 100 pairs, which form the chains of the
1-back lists, and that the sessions hold 20 pairs each.
"""
import argparse
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from analysis.helper import HUMAN_TASK_IDS  # noqa: E402
from mturk.helper import make_ordered_pair_df_with_features, select_balanced_chained_pair  # noqa: E402
from mturk.io import _subpath_after, localize_mturk_csv_filename  # noqa: E402
from mturk.task_configs import TASK_CONFIGS  # noqa: E402
from utils.helper import seed_everything  # noqa: E402
from utils.paths import TRIAL_DIR  # noqa: E402

A_MAP = {0: 'b', 1: 'x'}
PAIRS_PER_SESSION = 20
# TaskConfig feature -> pair-set column
FEATURE_COLUMN = {'cat_1b': 'same_category', 'obj': 'same_obj', 'pos_1b': 'same_pos'}
N_PRACTICE = {'1back': 2, 'dms': 3, 'interdms': 5, 'ctxdms': 8}


def get_nback_csv(trial_df, trials_per_session=PAIRS_PER_SESSION, act_map=A_MAP, features='same_category'):
    """1-back trials: each chain of 5 pairs (consecutive rows of `trial_df`) is one trial of 6 stimuli."""
    stim_per_trial = 6
    rows = []
    session = 0
    stim_chain = []
    act_chain = []
    for i, row in trial_df.iterrows():
        if i % trials_per_session == 0:
            session += 1
        if len(stim_chain) == 0:
            stim_chain.extend([row["stim1_fp"], row["stim2_fp"]])
        else:
            stim_chain.append(row["stim2_fp"])
        act_chain.append(row[features])

        if len(stim_chain) == stim_per_trial:
            rows.append({'session': session,
                         **{f'stim{k + 1}': stim_chain[k] for k in range(stim_per_trial)},
                         **{f'act{k + 2}': act_map[act_chain[k]] for k in range(stim_per_trial - 1)}})
            stim_chain = []
            act_chain = []
    return pd.DataFrame(rows)


def get_interdms_csv(trial_df, trials_per_session=PAIRS_PER_SESSION, act_map=A_MAP,
                     features=('same_category', 'same_category'), pattern='ABAB'):
    """InterDMS trials: pair A from one shuffle of `trial_df`, pair B from another, interleaved by `pattern`."""
    aa_df = trial_df.sample(frac=1).reset_index(drop=True)
    bb_df = trial_df.sample(frac=1).reset_index(drop=True)

    rows = []
    session = 0
    for i, ((_, aa), (_, bb)) in enumerate(zip(aa_df.iterrows(), bb_df.iterrows())):
        if i % trials_per_session == 0:
            session += 1
        row_d = {'session': session}
        for stim_i, which in enumerate(pattern):
            pair, feature = (aa, features[0]) if which == 'A' else (bb, features[1])
            if stim_i < 2:
                # the first stimulus of each pair
                row_d[f'stim{stim_i + 1}'] = pair['stim1_fp']
            else:
                row_d[f'stim{stim_i + 1}'] = pair['stim2_fp']
                row_d[f'act{stim_i + 1}'] = act_map[pair[feature]]
        rows.append(row_d)
    return pd.DataFrame(rows)


def get_dms_csv(trial_df, trials_per_session=PAIRS_PER_SESSION, act_map=A_MAP, features='same_category'):
    """DMS trials: every pair of `trial_df` once, in shuffled order."""
    trial_df = trial_df.sample(frac=1).reset_index(drop=True)
    rows = []
    session = 0
    for i, row in trial_df.iterrows():
        if i % trials_per_session == 0:
            session += 1
        rows.append({'session': session, 'stim1': row['stim1_fp'], 'stim2': row['stim2_fp'],
                     'act2': act_map[row[features]]})
    return pd.DataFrame(rows)


def get_ctx_dms_csv(trial_df, stim_df, trials_per_session=PAIRS_PER_SESSION, act_map=A_MAP,
                    features=('same_category', 'same_pos', 'same_obj')):
    """Context-DMS trials: a context stimulus, then every pair of `trial_df` in shuffled order.

    The context matches the pair's first stimulus in ``features[0]`` with probability 0.5 and is never one
    of the pair; the answer is the pair's match in ``features[1]`` if it does, else in ``features[2]``.
    """
    trial_df = trial_df.sample(frac=1).reset_index(drop=True)
    rows = []
    session = 0
    for i, row in trial_df.iterrows():
        if i % trials_per_session == 0:
            session += 1

        ctx_true = random.random() < 0.5
        ctx_feature = features[0]
        if 'category' in ctx_feature:
            same = stim_df['cat_1b'] == row['stim1_category']
        elif 'pos' in ctx_feature:
            same = stim_df['pos_1b'] == row['stim1_pos']
        elif 'obj' in ctx_feature:
            same = (stim_df['id_1b'] == row['stim1_obj']) & (stim_df['cat_1b'] == row['stim1_category'])
        else:
            raise ValueError(f"Unknown context feature: {ctx_feature}")
        filter_cond = same if ctx_true else ~same

        pair_indices = [row['stim1'], row['stim2']]
        context = stim_df[(filter_cond & ~stim_df.index.isin(pair_indices))].sample(n=1).iloc[0]
        response = row[features[1]] if ctx_true else row[features[2]]
        rows.append({'session': session, 'stim1': mturk_name(context['re_filename']), 'stim2': row['stim1_fp'],
                     'stim3': row['stim2_fp'], 'act3': act_map[response]})
    return pd.DataFrame(rows)


def mturk_name(filename):
    """images/<Variation folder>/<file name>, the name a stimulus was shown under (no leading underscore)."""
    return (Path('images') / _subpath_after(Path(filename), segment='HvM_with_discfade')).as_posix()


def load_subset(path):
    stim_df = pd.read_csv(path, index_col=0)
    return localize_mturk_csv_filename(stim_df)


def draw_pair_set(stim_df, n_tries):
    """100 ordered pairs in 20 chains of 5, with columns as make_ordered_pair_df_with_features."""
    pair_df = make_ordered_pair_df_with_features(stim_df)
    pair_set, _ = select_balanced_chained_pair(pair_df, n_trials=20, n_chains=5, n_tries_limit=n_tries)
    for col in ('stim1_fp', 'stim2_fp'):
        pair_set[col] = pair_set[col].map(mturk_name)
    return pair_set


def build_list(cfg, pair_set, stim_df):
    """The trial list of one task from a pair set."""
    kind = cfg.task_name.split('_')[0]
    columns = [FEATURE_COLUMN[f] for f in cfg.features]
    if kind == '1back':
        return get_nback_csv(pair_set, features=columns[0])
    if kind == 'dms':
        return get_dms_csv(pair_set, features=columns[0])
    if kind == 'interdms':
        return get_interdms_csv(pair_set, features=tuple(columns), pattern=cfg.task_name.split('_')[1])
    if kind == 'ctxdms':
        return get_ctx_dms_csv(pair_set, stim_df, features=tuple(columns))
    raise ValueError(cfg.task_name)


def build(out_dir, subset, seed, n_tries):
    seed_everything(seed)
    stim_df = load_subset(subset)
    main_set, practice_set = draw_pair_set(stim_df, n_tries), draw_pair_set(stim_df, n_tries)
    out_dir.mkdir(parents=True, exist_ok=True)
    if Path(subset).resolve() != (out_dir / 'stim_subset.csv').resolve():
        shutil.copy2(subset, out_dir / 'stim_subset.csv')
    for task_id in HUMAN_TASK_IDS:
        cfg = TASK_CONFIGS[task_id]
        practice = build_list(cfg, practice_set, stim_df).head(N_PRACTICE[cfg.task_name.split('_')[0]])
        trials = pd.concat([practice.assign(session=0), build_list(cfg, main_set, stim_df)], ignore_index=True)
        trials.to_csv(out_dir / cfg.task_trial_csv.name, index=False)
        print(f'{cfg.task_trial_csv.name}: {len(trials)} trials')


def _match(a, b, column):
    same_category = a['cat_1b'] == b['cat_1b']
    return int({'same_category': same_category,
                'same_obj': same_category and a['id_1b'] == b['id_1b'],
                'same_pos': a['pos_1b'] == b['pos_1b']}[column])


def check(trial_dir):
    """Check the structure of the trial lists in `trial_dir` (see the module docstring)."""
    stim_df = load_subset(trial_dir / 'stim_subset.csv')
    by_name = {Path(mturk_name(f)).name: r for f, (_, r) in zip(stim_df['re_filename'], stim_df.iterrows())}
    pair_sets = {}
    for task_id in HUMAN_TASK_IDS:
        cfg = TASK_CONFIGS[task_id]
        kind = cfg.task_name.split('_')[0]
        trials = pd.read_csv(trial_dir / cfg.task_trial_csv.name)
        columns = [FEATURE_COLUMN[f] for f in cfg.features]
        main_pairs = []
        for _, row in trials.iterrows():
            s = [by_name[Path(row[c]).name] for c in cfg.stim_cols]
            if kind == '1back':
                pairs = [(s[k], s[k + 1], columns[0], f'act{k + 2}') for k in range(5)]
            elif kind == 'dms':
                pairs = [(s[0], s[1], columns[0], 'act2')]
            elif kind == 'interdms':
                (a1, b1, x, y) = s
                if cfg.task_name.split('_')[1] == 'ABAB':
                    pairs = [(a1, x, columns[0], 'act3'), (b1, y, columns[1], 'act4')]
                else:
                    pairs = [(b1, x, columns[1], 'act3'), (a1, y, columns[0], 'act4')]
            else:
                ctx, first, second = s
                assert ctx.name not in (first.name, second.name), f'{cfg.task_name}: context is part of the pair'
                column = columns[1] if _match(ctx, first, columns[0]) else columns[2]
                pairs = [(first, second, column, 'act3')]
            for a, b, column, act in pairs:
                assert A_MAP[_match(a, b, column)] == row[act], f'{cfg.task_name}: wrong answer in {row.to_dict()}'
                if row['session'] != 0:
                    main_pairs.append((a.name, b.name))
        main = trials[trials['session'] != 0]
        per_session = main.groupby('session').size().to_dict()
        n_pairs_per_trial = 5 if kind == '1back' else 1
        assert per_session == {s: PAIRS_PER_SESSION // n_pairs_per_trial for s in range(1, 6)}, per_session
        pair_sets[cfg.task_name] = main_pairs
        print(f'{cfg.task_name}: {len(trials) - len(main)} practice and {len(main)} main trials, answers agree '
              f'with the stimulus metadata')

    chains = pair_sets['1back_position']
    assert len(chains) == 100 and len({tuple(sorted(p)) for p in chains}) == 100
    for name, pairs in pair_sets.items():
        counts = pd.Series(pairs).value_counts()
        assert set(counts.index) == set(chains), f'{name} compares other pairs than the 1-back chains'
        assert (counts == (2 if name.startswith('interdms') else 1)).all(), name
    rates = {c: np.mean([_match(stim_df.loc[a], stim_df.loc[b], c) for a, b in chains]) for c in FEATURE_COLUMN.values()}
    print('every list compares the same 100 pairs, the chains of the 1-back trials; match rates '
          + ', '.join(f'{c} {r:.2f}' for c, r in rates.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('out_dir', type=Path, nargs='?')
    parser.add_argument('--subset', type=Path, default=TRIAL_DIR / 'stim_subset.csv', help='stimulus subset')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--tries', type=int, default=10000, help='random draws per pair set')
    parser.add_argument('--check', type=Path, nargs='?', const=TRIAL_DIR, metavar='TRIAL_DIR',
                        help='check the trial lists in TRIAL_DIR instead of building new ones')
    args = parser.parse_args()
    if args.check:
        check(args.check)
    elif args.out_dir:
        build(args.out_dir, args.subset, args.seed, args.tries)
    else:
        parser.error('give OUT_DIR or --check')


if __name__ == '__main__':
    main()
