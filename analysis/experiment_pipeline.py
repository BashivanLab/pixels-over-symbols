"""Cached model, human and VLM responses on the MTurk trials, and the behavioral metrics of the paper.

:func:`run_experiment_pipeline` evaluates whatever is not yet cached in ``utils.paths.EVAL_DIR`` and
computes accuracy, consistency with humans, error consistency and the cross-task and cross-stimulus RSA;
:func:`embedding_pipeline` caches the hidden states that the DSA reads. Cache files of a model key, with
``<stem>`` = ``<key>_match<acc>`` for an accuracy-matched checkpoint and ``<key>`` otherwise:
    <stem>_conf.npy, <stem>_acc.npy   {'<key>_<seed>': (1800,)} confidence in the correct answer and
                                      correctness of every compared pair: 18 tasks (get_new_metadata) x
                                      the same 100 pairs, ordered as pair_keys.npy
    <stem>_acc_draws.npy              noisy models: {'<key>_<seed>': (n_draws, 1800)} correctness per
                                      noise draw
    <stem>_seq_feature.npy, <stem>_pre_feature.npy
                                      {seed: {task_id: (6 frames, trials of the task, hidden size)}} output
                                      of and input to the sequence core
A random-init twin ``<key>_rand`` has the same files under the stem ``<key>_rand``. Shared files:
human_responses.npy (participants x trials, correctness), feature_indices.npy, pair_keys.npy, and one
<vlm>.npy per VLM.
"""

import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from utils.paths import EVAL_DIR, HVM_DIR
from analysis.cross_task import get_cross_task_corr, get_cross_sequence_corr, compute_rsa_alignment
from analysis.eval_loop import eval_mturk
from analysis.factory import (
    load_all_experiment_models, load_emb_cache,
    NAT_CONFIG_DICT, AB_CONFIG_DICT, DIM_MATCH_CONFIG_DICT
)
from analysis.helper import (
    HUMAN_TASK_IDS,
    convert_eval_mturk_to_consistency_format, convert_human_responses_to_matrix,
    get_feature_indices, convert_task_features_to_matrix, fisher_mean_matrix, generate_subject_splits
)
from analysis.metrics.accuracy import accuracy_hierarchical
from analysis.metrics.consistency import consistency_hierarchical
from analysis.metrics.error_consistency import error_consistency_hierarchical
from mturk.stimuli import sample_df
from mturk.task_configs import validate_task_configs
from utils.const import TASK_METADATA
from utils.stim_io import PRETRAINED_STATS

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


def get_new_metadata():
    """Task ids of the metric arrays: the 12 MTurk tasks, each InterDMS task split in two.

    An InterDMS trial holds two comparisons (A and B), scored as two tasks with names suffixed _AA and
    _BB, so the 12 tasks give 18.

    Returns:
        ``(new_task_metadata, new_mapping)``: ``{new_task_id: {'name', 'type'}}`` and
        ``{new_task_id: original task id}``.
    """
    new_task_metadata = defaultdict(dict)
    new_mapping = dict()
    new_task_id = 1
    for task_id in HUMAN_TASK_IDS:
        orig_name = TASK_METADATA[task_id]['name']
        orig_type = TASK_METADATA[task_id]['type']
        if orig_type == 'InterDMS':
            for suffix in ['AA', 'BB']:
                new_task_metadata[new_task_id].update({'type': 'InterDMS', 'name': f'{orig_name}_{suffix}'})
                new_mapping[new_task_id] = task_id
                new_task_id += 1
        else:
            new_task_metadata[new_task_id].update({'name': orig_name, 'type': orig_type})
            new_mapping[new_task_id] = task_id
            new_task_id += 1
    return new_task_metadata, new_mapping


def cache_stem(key, cfg):
    """Stem of a model key's cache files: ``<key>_match<acc>`` for an accuracy-matched checkpoint, else the key."""
    return f"{key}_match{cfg['acc_match']}" if cfg['match_acc'] else key


def _draws_path(acc_path: Path) -> Path:
    """Cache path of a noisy model's per-draw correctness, next to its ``*_acc.npy``."""
    return acc_path.with_name(acc_path.name[:-len('_acc.npy')] + '_acc_draws.npy')


def _load_stimuli(hvm_dir, n_pairs):
    task_configs = validate_task_configs()
    img_loader, meta_data = sample_df(
        hvm_dir, 16, n_pairs, 3,
        df_path=task_configs[1].task_meta_csv,
        pretrained_stats=PRETRAINED_STATS['clip_resnet'],
    )
    return task_configs, img_loader, meta_data


def _active_configs(ab_model_keys, nat_model_keys, dim_match_model_keys,
                    custom_ab_configs, custom_nat_configs, custom_dim_match_configs):
    active_ab = {**AB_CONFIG_DICT, **(custom_ab_configs or {})}
    active_nat = {**NAT_CONFIG_DICT, **(custom_nat_configs or {})}
    active_dim_match = {**DIM_MATCH_CONFIG_DICT, **(custom_dim_match_configs or {})}
    assert set(ab_model_keys) <= set(active_ab.keys()), f'Missing {set(ab_model_keys) - set(active_ab.keys())}'
    assert set(nat_model_keys) <= set(active_nat.keys()), f'Missing {set(nat_model_keys) - set(active_nat.keys())}'
    assert set(dim_match_model_keys) <= set(active_dim_match.keys()), \
        f'Missing {set(dim_match_model_keys) - set(active_dim_match.keys())}'
    return {**active_nat, **active_ab, **active_dim_match}


def run_experiment_pipeline(
        experiment_name,
        ab_model_keys,
        nat_model_keys,
        vlm_model_keys=None,
        dim_match_model_keys=None,
        n_models=6,
        use_saved=True,
        device="cpu",
        hvm_dir=HVM_DIR,
        custom_nat_configs=None,
        custom_ab_configs=None,
        custom_dim_match_configs=None,
        n_pairs=100,
        n_splits=20,
        noise_draws=20,
        load_random=False,
):
    """Cached (or else evaluated) responses of the given models, humans and VLMs, and their metrics.

    Args:
        experiment_name: label for the log.
        ab_model_keys, nat_model_keys, dim_match_model_keys: keys of AB_CONFIG_DICT, NAT_CONFIG_DICT and
            DIM_MATCH_CONFIG_DICT, or of the custom configs merged over them.
        vlm_model_keys: VLM names (``utils.const.VLM_MODELS``).
        n_models: seeds per key.
        use_saved: read cached responses; if False, evaluate every model again.
        device: torch device of the evaluation.
        hvm_dir: HvM images and metadata.
        n_pairs: compared pairs per task.
        n_splits: split halves of participants and of model seeds for the consistency metrics.
        noise_draws: seeded noise draws per trial of noisy models; their confidence and accuracy are the
            means over draws and their error consistency the mean of the per-draw values.
        load_random: also load (or evaluate) the random-init twin ``<key>_rand`` of every key.

    Returns:
        Dict with ``all_model_conf``/``all_model_acc`` (``{key: {'<key>_<seed>': (n_trials,)}}``, VLMs
        included), ``all_model_acc_draws``, ``human_responses``, the task ids ``new_task_metadata`` and
        ``new_mapping``, the metrics ``cons_result``, ``acc_result``, ``ec_result`` (each
        ``{aggregation: ...}``) with their per-split samples, the ``model_splits``/``human_splits``,
        ``feat_index``, ``pair_keys``, ``rsa_results`` and the stimulus ``img_loader`` (None when
        everything came from the cache, which then needs no HvM data).

    The RSA draws its split halves from NumPy's global random state, model by model in the order of
    ``all_model_conf``, which starts with the requested keys in the iteration order of a set of
    strings. Both depend on the run: the paper's figures were made with PYTHONHASHSEED=0 and
    ``np.random.seed(0)`` before this call, from cached responses. The other metrics are deterministic.
    """
    agg_dir = EVAL_DIR
    agg_dir.mkdir(exist_ok=True)

    img_loader = None  # the HvM stimuli are only needed to evaluate what is not cached
    new_task_metadata, new_mapping = get_new_metadata()

    human_path = agg_dir / 'human_responses.npy'
    feat_index_path = agg_dir / 'feature_indices.npy'
    pair_keys_path = agg_dir / 'pair_keys.npy'

    dim_match_model_keys = dim_match_model_keys or []
    all_cfg = _active_configs(ab_model_keys, nat_model_keys, dim_match_model_keys,
                              custom_ab_configs, custom_nat_configs, custom_dim_match_configs)
    all_expected_keys = list(set(ab_model_keys + nat_model_keys + dim_match_model_keys))

    all_model_conf = dict()
    all_model_acc = dict()
    all_model_acc_draws = dict()  # noisy models: {key: {seed_id: (noise_draws, n_stimuli)}}
    missing_ab_keys = list()
    missing_nat_keys = list()
    missing_dim_match_keys = list()
    logger.info(f'Checking cache for {experiment_name}...')

    for model_key in all_expected_keys:
        cfg = all_cfg[model_key]
        stems = [cache_stem(model_key, cfg)] + ([f'{model_key}_rand'] if load_random else [])
        names = [model_key] + ([f'{model_key}_rand'] if load_random else [])
        noisy = cfg.get('noise_scale', 0) > 0
        paths = [agg_dir / f'{stem}_{kind}.npy' for stem in stems
                 for kind in ['conf', 'acc'] + (['acc_draws'] if noisy else [])]
        if use_saved and all(p.is_file() for p in paths):
            for name, stem in zip(names, stems):
                all_model_conf[name] = np.load(agg_dir / f'{stem}_conf.npy', allow_pickle=True).item()
                all_model_acc[name] = np.load(agg_dir / f'{stem}_acc.npy', allow_pickle=True).item()
            if noisy:
                for name, stem in zip(names, stems):
                    all_model_acc_draws[name] = np.load(agg_dir / f'{stem}_acc_draws.npy', allow_pickle=True).item()
        else:
            if model_key in ab_model_keys:
                missing_ab_keys.append(model_key)
            if model_key in nat_model_keys:
                missing_nat_keys.append(model_key)
            if model_key in dim_match_model_keys:
                missing_dim_match_keys.append(model_key)

    vlm_responses = dict()
    missing_vlm_keys = list()
    for model_key in vlm_model_keys or []:
        response_path = agg_dir / f'{model_key}.npy'
        if response_path.is_file():
            vlm_responses[model_key] = np.load(response_path, allow_pickle=True).item()
        else:
            missing_vlm_keys.append(model_key)

    # eval_mturk produces the human, VLM and model responses in one pass over the trials
    needs_human_eval = not (human_path.is_file() and feat_index_path.is_file() and pair_keys_path.is_file())
    if missing_ab_keys or missing_nat_keys or missing_dim_match_keys or needs_human_eval or missing_vlm_keys:
        logger.info(
            f'Cache missing, Computing fresh results for:\n'
            f'\tNatural: {missing_nat_keys}\n'
            f'\tAbstract: {missing_ab_keys}\n'
            f'\tDimMatch: {missing_dim_match_keys}\n'
            f'\tVLM: {missing_vlm_keys}\n'
        )

        task_configs, img_loader, meta_data = _load_stimuli(hvm_dir, n_pairs)
        emb_cache_dict = load_emb_cache(keys=missing_nat_keys, configs=custom_nat_configs)
        nat_models, ab_models = load_all_experiment_models(
            n_models=n_models,
            ab_model_keys=missing_ab_keys,
            nat_model_keys=missing_nat_keys,
            dim_match_model_keys=missing_dim_match_keys,
            custom_ab_configs=custom_ab_configs,
            custom_nat_configs=custom_nat_configs,
            custom_dim_match_configs=custom_dim_match_configs,
            load_random=load_random,
        )
        all_loaded_keys = list(ab_models.keys()) + list(nat_models.keys())
        logger.info(f'All models: {all_loaded_keys}')

        noise_draw_dfs = dict()
        _, task_pair_dfs, feature_pair_dfs, _ = eval_mturk(
            nat_models=nat_models,
            ab_models=ab_models,
            vlm_models=missing_vlm_keys,
            task_ids=HUMAN_TASK_IDS,
            img_loader=img_loader,
            meta_data=meta_data,
            task_configs=task_configs,
            device=torch.device(device),
            emb_cache_dict=emb_cache_dict,
            get_act=False,
            noise_draws=noise_draws,
            noise_draw_dfs=noise_draw_dfs,
        )

        for model_key in all_loaded_keys:
            stem = model_key if model_key.endswith('_rand') else cache_stem(model_key, all_cfg[model_key])
            conf_path = agg_dir / f'{stem}_conf.npy'
            acc_path = agg_dir / f'{stem}_acc.npy'
            conf = convert_eval_mturk_to_consistency_format(
                task_pair_dfs, model_type=model_key, metric='confs',
                aggregate_tasks=True
            )
            acc = convert_eval_mturk_to_consistency_format(
                task_pair_dfs, model_type=model_key, metric='acc',
                aggregate_tasks=True
            )
            # a key read from the cache keeps its cached arrays
            if model_key not in all_model_conf:
                all_model_conf[model_key] = conf
            if model_key not in all_model_acc:
                all_model_acc[model_key] = acc

            np.save(conf_path, conf, allow_pickle=True)
            np.save(acc_path, acc, allow_pickle=True)

            # noisy models: per-draw correctness, {seed_id: (n_draws, n_stimuli)}
            if any(model_key in per_task for per_task in noise_draw_dfs.values()):
                assert set(noise_draw_dfs) == set(task_pair_dfs), 'noise draws missing for some tasks'
                n_draws = len(next(iter(noise_draw_dfs.values()))[model_key][0])
                per_draw = [convert_eval_mturk_to_consistency_format(
                    {t: {model_key: [seed_draws[d] for seed_draws in noise_draw_dfs[t][model_key]]}
                     for t in noise_draw_dfs},
                    model_type=model_key, metric='acc', aggregate_tasks=True,
                ) for d in range(n_draws)]
                acc_draws = {s: np.stack([a[s] for a in per_draw]) for s in per_draw[0]}
                if model_key not in all_model_acc_draws:
                    all_model_acc_draws[model_key] = acc_draws
                np.save(_draws_path(acc_path), acc_draws, allow_pickle=True)
        if needs_human_eval:
            human_responses = convert_human_responses_to_matrix(task_pair_dfs, aggregate_tasks=True)
            np.save(human_path, human_responses, allow_pickle=True)

            _, feat_index, _ = get_feature_indices(
                feature_pair_dfs,
                task_pair_dfs,
                task_configs,
                new_task_metadata,
                new_mapping,
                n_pairs=100,
            )
            np.save(feat_index_path, feat_index, allow_pickle=True)

            # the pairs of task 1 in the order of every response vector (sorted pair keys)
            pair_keys = np.array(task_pair_dfs[1]['human'][0].sort_values('pair_key')['pair_key'].tolist())
            np.save(pair_keys_path, pair_keys, allow_pickle=True)
        else:
            human_responses = np.load(human_path, allow_pickle=True)
            feat_index = np.load(feat_index_path, allow_pickle=True).item()
            pair_keys = np.load(pair_keys_path, allow_pickle=True)

        for model_key in missing_vlm_keys:
            vlm_response = convert_eval_mturk_to_consistency_format(
                task_pair_dfs, model_type=model_key, metric='acc',
                aggregate_tasks=True
            )
            np.save(agg_dir / f'{model_key}.npy', vlm_response, allow_pickle=True)
            vlm_responses[model_key] = vlm_response
    else:
        logger.info('All requested models and human data successfully loaded from cache.')
        human_responses = np.load(human_path, allow_pickle=True)
        feat_index = np.load(feat_index_path, allow_pickle=True).item()
        pair_keys = np.load(pair_keys_path, allow_pickle=True)

    model_splits = generate_subject_splits(n_subjects=n_models, n_splits=n_splits, random_state=42)
    human_splits = generate_subject_splits(n_subjects=20, n_splits=n_splits, random_state=42)

    if vlm_model_keys is not None:
        # a VLM run gives one answer per trial, so its correctness serves as both confidence and accuracy
        all_model_conf.update(vlm_responses)
        all_model_acc.update(vlm_responses)

    flat_feat_index = {
        feat: np.concatenate(list(task_indices.values()), axis=0).astype(int)
        for feat, task_indices in feat_index.items()
    }
    metric_aggs = ('per_task', 'per_task_type', 'per_feature', 'overall')
    cons_result, cons_samples = consistency_hierarchical(
        all_model_outputs=all_model_conf, human_responses=human_responses,
        n_pairs=n_pairs, task_metadata=new_task_metadata,
        model_splits=model_splits, human_splits=human_splits,
        aggs=metric_aggs, feature_indices=flat_feat_index,
    )
    acc_result, acc_samples = accuracy_hierarchical(
        all_model_outputs=all_model_acc, human_responses=human_responses,
        n_pairs=n_pairs, task_metadata=new_task_metadata,
        aggs=metric_aggs, feature_indices=flat_feat_index,
    )
    ec_result, ec_samples = error_consistency_hierarchical(
        all_model_acc=all_model_acc, human_responses=human_responses,
        n_pairs=n_pairs, n_models=n_models, task_metadata=new_task_metadata,
        aggs=metric_aggs, feature_indices=flat_feat_index, acc_draws=all_model_acc_draws,
    )

    # Cross-task and cross-stimulus correlation matrices (split-half, averaged in Fisher z over splits)
    # and their alignment with the human ones. The responses are task-major: (n_tasks, n_pairs) per
    # participant or seed.
    n_tasks = len(new_task_metadata)
    human_data_3d = human_responses.reshape(-1, n_tasks, n_pairs)
    raw_task_matrix = get_cross_task_corr(human_data_3d)
    raw_seq_matrix = get_cross_sequence_corr(human_data_3d)
    human_cross_task_matrix = fisher_mean_matrix(raw_task_matrix, axis=2)
    human_cross_seq_matrix = fisher_mean_matrix(raw_seq_matrix, axis=2)
    rsa_results = {
        'human_cross_task': human_cross_task_matrix,
        'human_cross_seq': human_cross_seq_matrix,

        'model_cross_task': {},
        'model_cross_seq': {},
        'task_rsa_alignment': {'models': {}},
        'seq_rsa_alignment': {'models': {}},
    }

    for m_key, m_out in all_model_conf.items():
        if m_key.endswith('_rand'):
            continue

        m_vals = np.array(list(m_out.values()))  # (n_seeds, n_trials)
        m_data_3d = m_vals.reshape(-1, n_tasks, n_pairs)

        raw_task_matrix = get_cross_task_corr(m_data_3d)
        m_cross_task_matrix = fisher_mean_matrix(raw_task_matrix, axis=2)
        rsa_results['model_cross_task'][m_key] = m_cross_task_matrix

        res = compute_rsa_alignment(human_cross_task_matrix, m_cross_task_matrix)
        rsa_results['task_rsa_alignment']['models'][m_key] = {'mean': res}

        raw_seq_matrix = get_cross_sequence_corr(m_data_3d)
        m_cross_seq_matrix = fisher_mean_matrix(raw_seq_matrix, axis=2)
        rsa_results['model_cross_seq'][m_key] = m_cross_seq_matrix
        res = compute_rsa_alignment(human_cross_seq_matrix, m_cross_seq_matrix)
        rsa_results['seq_rsa_alignment']['models'][m_key] = {'mean': res}

    return {
        'all_model_conf': all_model_conf,
        'all_model_acc': all_model_acc,
        'all_model_acc_draws': all_model_acc_draws,
        'human_responses': human_responses,
        'new_task_metadata': new_task_metadata,
        'new_mapping': new_mapping,
        'cons_result': cons_result, 'cons_samples': cons_samples,
        'acc_result': acc_result, 'acc_samples': acc_samples,
        'ec_result': ec_result, 'ec_samples': ec_samples,
        'model_splits': model_splits,
        'human_splits': human_splits,
        'feat_index': feat_index,
        'pair_keys': pair_keys,
        'rsa_results': rsa_results,
        'img_loader': img_loader
    }


def embedding_pipeline(
        experiment_name,
        ab_model_keys,
        nat_model_keys,
        n_models=6,
        use_saved=True,
        device="cpu",
        hvm_dir=HVM_DIR,
        custom_nat_configs=None,
        custom_ab_configs=None,
        custom_dim_match_configs=None,
        dim_match_model_keys=None,
        n_pairs=100,
        load_random=False,
):
    """Cached (or else computed) hidden states of the given models on the MTurk trials.

    For every model and seed it stores, per task, the output of the sequence core (``seq_feature``) and
    its input after the input pathway (``pre_feature``), each ``(T, n_trials, hidden)``. Noisy models run
    one seeded noise draw. Arguments as in :func:`run_experiment_pipeline`.

    Returns:
        Dict with ``all_model_seq_feature`` and ``all_model_pre_feature``
        (``{key: {seed: {task_id: array}}}``), ``new_task_metadata``, ``new_mapping`` and ``img_loader``
        (None when every feature came from the cache).
    """
    agg_dir = EVAL_DIR
    agg_dir.mkdir(exist_ok=True)

    img_loader = None
    new_task_metadata, new_mapping = get_new_metadata()

    dim_match_model_keys = dim_match_model_keys or []
    all_cfg = _active_configs(ab_model_keys, nat_model_keys, dim_match_model_keys,
                              custom_ab_configs, custom_nat_configs, custom_dim_match_configs)
    all_expected_keys = list(set(ab_model_keys + nat_model_keys + dim_match_model_keys))

    all_model_seq_feature = dict()
    all_model_pre_feature = dict()
    missing_ab_keys = list()
    missing_nat_keys = list()
    missing_dim_match_keys = list()
    logger.info(f'Checking cache for {experiment_name}...')

    for model_key in all_expected_keys:
        names = [model_key] + ([f'{model_key}_rand'] if load_random else [])
        stems = [cache_stem(model_key, all_cfg[model_key])] + names[1:]
        if use_saved and all((agg_dir / f'{stem}_{kind}_feature.npy').is_file() for stem in stems
                             for kind in ('seq', 'pre')):
            for name, stem in zip(names, stems):
                all_model_seq_feature[name] = np.load(agg_dir / f'{stem}_seq_feature.npy', allow_pickle=True).item()
                all_model_pre_feature[name] = np.load(agg_dir / f'{stem}_pre_feature.npy', allow_pickle=True).item()
        else:
            if model_key in ab_model_keys:
                missing_ab_keys.append(model_key)
            if model_key in nat_model_keys:
                missing_nat_keys.append(model_key)
            if model_key in dim_match_model_keys:
                missing_dim_match_keys.append(model_key)

    if missing_ab_keys or missing_nat_keys or missing_dim_match_keys:
        logger.info(
            f'Cache missing, Computing fresh results for:\n'
            f'\tNatural: {missing_nat_keys}\n'
            f'\tAbstract: {missing_ab_keys}\n'
            f'\tDimMatch: {missing_dim_match_keys}\n'
        )

        task_configs, img_loader, meta_data = _load_stimuli(hvm_dir, n_pairs)
        nat_models, ab_models = load_all_experiment_models(
            n_models=n_models,
            ab_model_keys=missing_ab_keys,
            nat_model_keys=missing_nat_keys,
            dim_match_model_keys=missing_dim_match_keys,
            custom_ab_configs=custom_ab_configs,
            custom_nat_configs=custom_nat_configs,
            custom_dim_match_configs=custom_dim_match_configs,
            load_random=load_random,
        )
        all_loaded_keys = list(ab_models.keys()) + list(nat_models.keys())
        _, task_pair_dfs, feature_pair_dfs, task_features = eval_mturk(
            nat_models=nat_models,
            ab_models=ab_models,
            vlm_models=list(),
            task_ids=HUMAN_TASK_IDS,
            img_loader=img_loader,
            meta_data=meta_data,
            task_configs=task_configs,
            device=torch.device(device),
            emb_cache_dict=load_emb_cache(keys=missing_nat_keys, configs=custom_nat_configs),
            get_act=True,
        )

        for model_key in all_loaded_keys:
            logger.info(f'Saving features for {model_key}')
            base_key = model_key.removesuffix('_rand')
            model_seq_hidden, model_pre_hidden = convert_task_features_to_matrix(
                task_features=task_features,
                model_type=model_key,
                has_visual_features=base_key in nat_model_keys,
            )
            stem = model_key if model_key.endswith('_rand') else cache_stem(model_key, all_cfg[model_key])
            if model_key not in all_model_seq_feature:
                all_model_seq_feature[model_key] = model_seq_hidden
            if model_key not in all_model_pre_feature:
                all_model_pre_feature[model_key] = model_pre_hidden
            np.save(agg_dir / f'{stem}_seq_feature.npy', model_seq_hidden, allow_pickle=True)
            np.save(agg_dir / f'{stem}_pre_feature.npy', model_pre_hidden, allow_pickle=True)
    else:
        logger.info('All requested model features successfully loaded from cache.')

    return {
        'all_model_seq_feature': all_model_seq_feature,
        'all_model_pre_feature': all_model_pre_feature,
        'new_task_metadata': new_task_metadata,
        'new_mapping': new_mapping,
        'img_loader': img_loader
    }
