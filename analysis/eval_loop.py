"""Responses of models, humans and VLMs on the MTurk trials.

``eval_mturk`` runs every model seed on the trials the participants saw and scores the human and VLM
responses the same way, one row per compared stimulus pair.
"""

from typing import Dict, List

import numpy as np
import pandas as pd
import torch

from utils.paths import HUMAN_DIR
from models.model import CNNRNN, CustomRNN
from mturk.responses import get_human_responses, process_human_responses
from mturk.task_configs import TaskConfig, validate_task_configs
from mturk.trials import mturk_trial_to_batches, task_info_to_one_hot
from mturk.vlm.vlm_loader import get_vlm_responses
from utils.const import TASK_NAME_TASK_INDEX
from utils.logger import SummaryLogger
from utils.stim_io import HvMImageLoader, HvMMetaData, HvMImageMapper

# context-DMS tasks: their pair keys hold the context stimulus first
CTXDMS_TASK_IDS = [37, 38, 39, 40]


def trial_noise_seeds(model_seed: int, task_id: int, n_trials: int, draw: int) -> List[int]:
    """Input-noise seed of each trial of a task, for one noise draw of a noisy model.

    Seeded from (model seed, task, trial, draw), so every prediction is reproducible; the model seed is
    the model's index in its key's list.
    """
    return [int(np.random.SeedSequence([model_seed, task_id, i, draw]).generate_state(1)[0])
            for i in range(n_trials)]


def mean_over_draws(dfs: List[pd.DataFrame]) -> pd.DataFrame:
    """Pair dataframe whose per-trial accuracy and confidence are averaged over noise draws.

    ``dfs`` are ``SummaryLogger.results_to_pair_df`` outputs for the same trials, so their rows line up.
    Accuracy becomes the fraction of draws on which the trial was correct (not a threshold on the
    averaged confidence).
    """
    out = dfs[0].copy()
    for d in dfs[1:]:
        assert d['pair_key'].tolist() == out['pair_key'].tolist(), 'noise draws disagree on pair order'
    for col in ('acc', 'confs'):
        out[col] = [np.mean(np.stack(vals), axis=0) for vals in zip(*(d[col] for d in dfs))]
    for col in ('mean_acc', 'mean_conf'):
        out[col] = np.mean(np.stack([d[col].to_numpy(dtype=float) for d in dfs]), axis=0)
    return out


def _decision_pair_keys(pair_df_list):
    """Drop the context stimulus from the pair keys of context-DMS trials.

    The pair keys keep the order of presentation (context, first compared, second compared), so
    ``key[1:]`` is the compared pair, which makes the keys comparable across trials with different
    context stimuli and with the two-stimulus keys of the other tasks.
    """
    for pair_df in pair_df_list:
        pair_df['pair_key_old'] = pair_df['pair_key']
        pair_df['pair_key'] = pair_df['pair_key_old'].apply(lambda x: tuple(x[1:]))


@torch.no_grad()
def eval_mturk(
        nat_models: Dict[str, List[CNNRNN]],
        ab_models: Dict[str, List[CustomRNN]],
        vlm_models: List[str],
        task_ids: List[int],
        img_loader: HvMImageLoader,
        meta_data: HvMMetaData,
        task_configs: Dict[int, TaskConfig] = None,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        ignore_learning: bool = True,
        pad_len=6,
        emb_cache_dict: Dict[str, tuple] = None,
        get_act: bool = False,
        noise_draws: int = 20,
        noise_draw_dfs: Dict = None,
):
    """Evaluate models, humans and VLMs on the MTurk trials of the tasks in `task_ids`.

    Args:
        nat_models: natural models ``{key: [seed models]}``; they read cached encoder features from
            ``emb_cache_dict[key]`` (``factory.load_emb_cache``), and a random-init twin
            ``'<key>_rand'`` reads those of ``key``.
        ab_models: one-hot abstract and dimensionality-matched models ``{key: [seed models]}``.
        vlm_models: VLM names (``utils.const.VLM_MODELS``).
        task_ids: the tasks to evaluate, of those in ``task_configs``.
        get_act: also return the input to the sequence core and its hidden states per frame.
        noise_draws: noisy models (``noise_scale > 0``) run this many seeded noise draws per trial and
            seed, and their pair dataframes hold the means over draws; with ``get_act`` they run draw 0
            only, since an average of noisy hidden-state trajectories is not a trajectory.
        noise_draw_dfs: if a dict, ``noise_draw_dfs[task_id][key][seed]`` receives the task dataframe of
            every draw of a noisy model (error consistency needs the correctness of each draw).

    Returns:
        ``(task_results, task_pair_dfs, feature_pair_dfs, task_features)``, each
        ``{task_id: {responder: [one entry per seed or participant]}}``, the responders being 'human',
        the VLMs and the model keys; the pair dataframes have one row per compared stimulus pair (an
        InterDMS trial gives two, with task ids '<id>_AA' and '<id>_BB').
    """
    if task_configs is None:
        task_configs = validate_task_configs()
    for model_list in list(nat_models.values()) + list(ab_models.values()):
        for model in model_list:
            model.eval()

    # deterministic kernels, so repeated evaluations give identical outputs
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    mapper = HvMImageMapper(img_loader=img_loader)
    logger = SummaryLogger(
        hvm_loader_df=img_loader.df,
        mapper=mapper,
        accumulate=False,
        # keep the stimuli of each pair key in order of presentation (see _decision_pair_keys)
        sort_pair=False,
    )

    task_results = dict()
    task_pair_dfs = dict()
    feature_pair_dfs = dict()
    task_features = dict()
    for task_id, task_config in task_configs.items():
        if task_id not in task_ids:
            continue
        print(f"========= Task {task_id} =========")

        def default_dict():
            d = {'human': list()}
            d.update({model_name: list() for model_name in vlm_models})
            d.update({model_name: list() for model_name in ab_models.keys()})
            d.update({model_name: list() for model_name in nat_models.keys()})
            return d

        task_results[task_id] = default_dict()
        task_pair_dfs[task_id] = default_dict()
        feature_pair_dfs[task_id] = default_dict()
        task_features[task_id] = default_dict()

        # number of stimuli per trial (2 for DMS, 3 for context DMS, 4 for InterDMS, 6 for 1-back)
        trial_len = len(task_config.stim_cols)
        trial_df = pd.read_csv(task_config.task_trial_csv)
        _, labels, task_info, _ = mturk_trial_to_batches(
            trial_df, img_loader, meta_data, trial_len=pad_len, ignore_learning=ignore_learning,
            load_images=False,
        )
        human_detail_dfs = get_human_responses(
            task_config=task_config,
            behaviour_dir=HUMAN_DIR / task_config.task_name,
            ignore_learning=ignore_learning
        )
        if len(human_detail_dfs) == 0:
            raise RuntimeError(f'Found no response csv for task {task_id} in {HUMAN_DIR}')
        per_subject_out = process_human_responses(
            dfs=human_detail_dfs,
            trial_len=trial_len,
            pad_len=pad_len
        )

        task_id = task_config.task_id
        assert task_id == TASK_NAME_TASK_INDEX[task_config.task_name]
        # the task one-hot of every trial (task ids are 1-based)
        task_index = torch.zeros((labels.shape[0], len(TASK_NAME_TASK_INDEX)))
        task_index[:, task_id - 1] = 1.0

        input_one_hot = task_info_to_one_hot(task_info, trial_len=pad_len)
        for nat_model_name, nat_model_list in nat_models.items():
            emb_cache, blank_cache = emb_cache_dict[nat_model_name.removesuffix('_rand')]
            input_emb, _, _, _ = mturk_trial_to_batches(
                trial_df, img_loader, meta_data, trial_len=pad_len, ignore_learning=ignore_learning,
                emb_cache=emb_cache, blank_cache=blank_cache,
            )
            for nat_model in nat_model_list:
                # one seed on the device at a time
                nat_model.to(device)
                outputs, _, features = nat_model(
                    input_emb.to(device),
                    task_index.to(device),
                    get_act=get_act,
                )
                outputs = outputs.to('cpu')
                if get_act:
                    # the encoder features ('cnn') are dropped: nothing reads them, and keeping them for
                    # every model, seed and task can exhaust memory
                    features = (None, features['pre'], features['post'])

                batch_results = logger.batch_update(
                    labels=labels,
                    outputs=outputs,
                    task_info=task_info,
                    task_index=task_index
                )
                nat_model.to('cpu')

                df_pair_tasks, df_pair_features, _ = logger.results_to_pair_df(batch_results, split_interdms=True)
                task_results[task_id][nat_model_name].append(batch_results)
                task_pair_dfs[task_id][nat_model_name].append(df_pair_tasks)
                feature_pair_dfs[task_id][nat_model_name].append(df_pair_features)
                task_features[task_id][nat_model_name].append(features)
        torch.cuda.empty_cache()

        # One-hot input; noisy models get seeded per-trial noise, noise_draws draws whose confidence and
        # correctness are averaged
        for ab_model_name, ab_model_list in ab_models.items():
            for model_seed, ab_model in enumerate(ab_model_list):
                ab_model.to(device)
                noisy = getattr(ab_model, 'noise_scale', 0) > 0
                n_draws = noise_draws if noisy and not get_act else 1
                draws = []
                for draw in range(n_draws):
                    noise_kw = {'noise_seeds': trial_noise_seeds(
                        model_seed, task_id, input_one_hot.shape[0], draw)} if noisy else {}
                    outputs, _, features = ab_model(
                        input_one_hot.to(device),
                        task_index.to(device),
                        get_act=get_act,
                        **noise_kw
                    )
                    outputs = outputs.to('cpu')
                    batch_results = logger.batch_update(
                        labels=labels,
                        outputs=outputs,
                        task_info=task_info,
                        task_index=task_index
                    )
                    draws.append((batch_results, *logger.results_to_pair_df(batch_results, split_interdms=True)[:2]))
                ab_model.to('cpu')

                if get_act:
                    features = (features['pre'], features['post'])

                batch_results, df_pair_tasks, df_pair_features = draws[0]
                if n_draws > 1:
                    df_pair_tasks = mean_over_draws([d[1] for d in draws])
                    df_pair_features = mean_over_draws([d[2] for d in draws])
                if noisy and not get_act and noise_draw_dfs is not None:
                    # copies: with a single draw, draws[0] is also the dataframe kept below
                    noise_draw_dfs.setdefault(task_id, {}).setdefault(ab_model_name, []).append(
                        [d[1].copy() for d in draws])
                task_results[task_id][ab_model_name].append(batch_results)
                task_pair_dfs[task_id][ab_model_name].append(df_pair_tasks)
                feature_pair_dfs[task_id][ab_model_name].append(df_pair_features)
                task_features[task_id][ab_model_name].append(features)
        torch.cuda.empty_cache()

        # human and VLM responses are already one-hot probabilities, so no softmax
        for worker_out in per_subject_out:
            outputs = worker_out.to(device)
            batch_results = logger.batch_update(
                labels=labels,
                outputs=outputs,
                task_info=task_info,
                task_index=task_index,
                outputs_are_probs=True
            )
            df_pair_tasks, df_pair_features, _ = logger.results_to_pair_df(batch_results, split_interdms=True)
            task_results[task_id]['human'].append(batch_results)
            task_pair_dfs[task_id]['human'].append(df_pair_tasks)
            feature_pair_dfs[task_id]['human'].append(df_pair_features)

        for model_name in vlm_models:
            # six runs of each VLM, in place of the six model seeds
            for probs in get_vlm_responses(task_config.task_name, model_name, pad_len=pad_len,
                                           ignore_learning=ignore_learning):
                batch_results = logger.batch_update(
                    labels=labels,
                    outputs=probs.to(device),
                    task_info=task_info,
                    task_index=task_index,
                    outputs_are_probs=True,
                )
                df_pair_tasks, df_pair_features, _ = logger.results_to_pair_df(batch_results, split_interdms=True)
                task_results[task_id][model_name].append(batch_results)
                task_pair_dfs[task_id][model_name].append(df_pair_tasks)
                feature_pair_dfs[task_id][model_name].append(df_pair_features)

    for task_id in CTXDMS_TASK_IDS:
        if task_id in task_pair_dfs:
            for pair_df_list in task_pair_dfs[task_id].values():
                _decision_pair_keys(pair_df_list)
            for pair_df_list in feature_pair_dfs[task_id].values():
                _decision_pair_keys(pair_df_list)
        if noise_draw_dfs is not None:
            for per_seed in noise_draw_dfs.get(task_id, {}).values():
                for seed_draws in per_seed:
                    _decision_pair_keys(seed_draws)

    return task_results, task_pair_dfs, feature_pair_dfs, task_features
