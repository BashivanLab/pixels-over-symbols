"""Visual-difficulty covariate control for the behavioral-consistency analyses.

Estimates a per-trial perceptual discriminability covariate from vision-encoder
embeddings of the two compared stimuli, residualizes the human and model
response vectors on it (per task, on that covariate alone), and recomputes the
consistency metrics on the residuals. If the Natural models' human-consistency
advantage survives this regression, it cannot be explained by shared stimulus
difficulty alone.

The covariate can come from the cached backbone features the Natural models
consume (``analysis.factory.load_emb_cache``) or, for an independence check,
from any encoder outside the model battery: generate its cache first with
``cache_image_embed.py --cnn_type <timm_name>`` and load the resulting
``*_features.pt``.

Usage (after ``run_experiment_pipeline`` and a fresh/kept ``task_pair_dfs``
from ``analysis.eval_loop.eval_mturk`` -- a human-only call with empty model
dicts is enough). Note that ``res['feat_index']`` is the *nested*
``{feature: {task_id: idx}}`` second return of ``get_feature_indices``; both
functions here want the flat first return, so take it directly::

    flat_fi, _, _ = get_feature_indices(feature_pair_dfs, task_pair_dfs,
                                        task_configs, res['new_task_metadata'],
                                        res['new_mapping'], n_pairs=100)
    info = build_global_pair_info(task_pair_dfs, res['img_loader'], flat_fi)
    emb_cache, _ = load_emb_cache(keys=[key], configs=None)[key]
    sim = pair_similarity(info['pairs'], res['img_loader'], emb_cache)
    cons_res, cons_samples, diag = difficulty_regressed_consistency(
        res['all_model_conf'], res['human_responses'], sim, info['match'],
        res['new_task_metadata'], res['model_splits'], res['human_splits'],
        feature_indices=flat_fi,
    )

Read the effect against ``shuffled_covariate_null``, not against the
unregressed baseline -- see that function for why.
"""

import numpy as np
import torch

from analysis.helper import _process_task_dataframe
from analysis.metrics.consistency import consistency_hierarchical


def build_global_pair_info(task_pair_dfs, img_loader, feature_indices, n_pairs=100):
    """Recover, per global trial index, the stimulus pair and its ground truth.

    Replays the exact concatenation order used by
    ``convert_human_responses_to_matrix``/``convert_eval_mturk_to_consistency_format``
    (original task ids sorted, InterDMS split into AA/BB, rows sorted by
    ``pair_key``), so position ``i`` here matches position ``i`` of every
    response vector.

    Args:
        task_pair_dfs: Output of ``analysis.eval_loop.eval_mturk`` (the
            ``'human'`` entries are sufficient, so an eval with empty model
            dicts works).
        img_loader: The shared ``HvMImageLoader`` from ``sample_df`` (provides
            per-row ``cat_1b``/``id_1b``/``pos_1b`` for the match labels).
        feature_indices: ``{'category'|'identity'|'position' -> global indices}``
            (first return of ``analysis.helper.get_feature_indices``); gives the
            compared feature of each trial.
        n_pairs: Stimulus pairs per (renumbered) task.

    Returns:
        Dict with:
        - ``'pairs'``: (N, 2) int array of ``img_loader.df`` row indices.
        - ``'feature'``: (N,) object array, compared feature per trial.
        - ``'match'``: (N,) bool array, ground-truth match per trial.
    """
    all_pairs = []
    for task_id in sorted(task_pair_dfs.keys()):
        df = task_pair_dfs[task_id]['human'][0]
        vectors = _process_task_dataframe(
            df, task_id, metric='pair_key', split_interdms=True, cast_to_int=False
        )
        for v in vectors:
            assert len(v) == n_pairs, \
                f'Task {task_id}: expected {n_pairs} pairs, found {len(v)}'
            all_pairs.extend(v)

    pairs = np.array([tuple(int(r) for r in p) for p in all_pairs])
    assert pairs.shape[1] == 2, 'Every pair_key must reduce to 2 stimulus rows'
    n_total = pairs.shape[0]

    # invert feature_indices: global index -> compared feature
    feature = np.empty(n_total, dtype=object)
    for f, idx in feature_indices.items():
        feature[np.asarray(idx, dtype=int)] = f
    assert not (feature == None).any(), 'feature_indices do not cover all trials'  # noqa: E711

    # ground-truth match from the stimulus metadata; identity means same object,
    # i.e. same (category, identity) since id_1b is only unique within category
    df = img_loader.df
    cat = df['cat_1b'].to_numpy()[pairs]
    obj = df['id_1b'].to_numpy()[pairs]
    pos = df['pos_1b'].to_numpy()[pairs]
    same_cat = cat[:, 0] == cat[:, 1]
    match = np.where(
        feature == 'category', same_cat,
        np.where(
            feature == 'identity', same_cat & (obj[:, 0] == obj[:, 1]),
            pos[:, 0] == pos[:, 1]
        )
    ).astype(bool)
    return {'pairs': pairs, 'feature': feature, 'match': match}


def pair_similarity(pairs, img_loader, emb_cache, pool='flatten'):
    """Cosine similarity between the two compared stimuli of every trial.

    Args:
        pairs: (N, 2) array of ``img_loader.df`` row indices
            (``build_global_pair_info``'s ``'pairs'``).
        img_loader: The same loader; its task cache's ``filename_to_idx`` maps
            each row's filename to its index in ``emb_cache`` (the same lookup
            ``mturk_trial_to_batches`` uses to feed the Natural models).
        emb_cache: (N_images, ...) feature tensor from ``load_emb_cache`` or a
            ``cache_image_embed.py``-generated ``*_features.pt``.
        pool: 'flatten' keeps the spatial/token structure (position-relevant),
            'mean' averages tokens/patches to a single vector.

    Returns:
        (N,) float array of cosine similarities.
    """
    filename_to_idx = img_loader._task_cache.filename_to_idx
    df = img_loader.df

    rows = np.unique(pairs)
    feats = {}
    for r in rows:
        cache_idx = filename_to_idx[str(df.loc[r, 'filename'])]
        assert cache_idx < len(emb_cache), \
            f'Row {r} maps to cache index {cache_idx} outside emb_cache ({len(emb_cache)})'
        f = emb_cache[cache_idx].float()
        if pool == 'mean' and f.dim() > 1:
            f = f.reshape(-1, f.shape[-1]).mean(dim=0)
        feats[r] = f.reshape(-1)
    feat_mat = torch.stack([feats[r] for r in rows])
    feat_mat = feat_mat / feat_mat.norm(dim=1, keepdim=True)
    row_pos = {r: i for i, r in enumerate(rows)}

    a = feat_mat[[row_pos[r] for r in pairs[:, 0]]]
    b = feat_mat[[row_pos[r] for r in pairs[:, 1]]]
    return (a * b).sum(dim=1).numpy().astype(np.float64)


def _within_type_difficulty(sim_block, match_block):
    """Signed, within-trial-type-centred difficulty covariate for one task block.

    Similarity predicts accuracy in opposite directions on match and non-match
    trials (a similar pair is easy to call "same", hard to call "different"), so
    the covariate is signed ``+sim`` / ``-sim``. But ``sim`` never approaches
    zero (observed range ~[0.44, 0.65]), so a plain signed covariate is ~0.997
    correlated with the trial-type indicator and regressing on it would strip
    out the match/non-match accuracy difference -- task structure, not visual
    difficulty. Centring within each trial type first makes the covariate
    exactly orthogonal to trial type, so the projection can only remove
    variation in stimulus confusability.
    """
    d = np.zeros_like(sim_block, dtype=np.float64)
    for is_match, sign in ((True, 1.0), (False, -1.0)):
        idx = (match_block == is_match)
        if idx.sum() > 0:
            d[idx] = (sim_block[idx] - sim_block[idx].mean()) * sign
    return d


def residualize_on_difficulty(human_responses, all_model_outputs, similarity, match, n_pairs=100):
    """OLS-residualize responses on the difficulty covariate, per task block.

    The design within each task's ``n_pairs`` trials is ``[1, difficulty]``:
    one intercept and one slope, so the projection removes what visual
    difficulty predicts and nothing else. The hat matrix is a fixed linear
    projector per block, so residualization commutes with the split-half
    subject/seed averaging inside the consistency metrics.

    ``difficulty`` is built by :func:`_within_type_difficulty`: signed by trial
    type to respect the direction flip, then centred within each trial type so
    it is orthogonal to the match/non-match indicator. The projection therefore
    removes variation in stimulus confusability and leaves the match/non-match
    accuracy difference, which is task structure rather than visual difficulty,
    untouched.

    Args:
        human_responses: (n_subjects, N) response matrix.
        all_model_outputs: ``{model_key: {seed_id: (N,) vector}}``.
        similarity: (N,) covariate from ``pair_similarity``.
        match: (N,) bool ground-truth labels from ``build_global_pair_info``,
            used only to sign the covariate.
        n_pairs: Trials per task block.

    Returns:
        ``(human_resid, model_resid, diagnostics)`` where diagnostics holds the
        per-block R^2 of the covariate on the subject-mean human response.
    """
    similarity = np.asarray(similarity, dtype=np.float64)
    match = np.asarray(match, dtype=bool)
    n_total = similarity.shape[0]
    assert human_responses.shape[1] == n_total
    assert n_total % n_pairs == 0
    n_blocks = n_total // n_pairs

    human_resid = np.asarray(human_responses, dtype=np.float64).copy()
    model_resid = {
        m: {s: np.asarray(v, dtype=np.float64).copy() for s, v in seeds.items()}
        for m, seeds in all_model_outputs.items()
    }
    r2_blocks = np.zeros(n_blocks)

    for b in range(n_blocks):
        sl = slice(b * n_pairs, (b + 1) * n_pairs)
        d = _within_type_difficulty(similarity[sl], match[sl])
        X = np.column_stack([np.ones_like(d), d])
        # pinv handles degenerate blocks (e.g. a constant covariate)
        hat = X @ np.linalg.pinv(X)

        h_block = human_resid[:, sl]
        human_resid[:, sl] = h_block - h_block @ hat.T
        for seeds in model_resid.values():
            for seed_id, v in seeds.items():
                seeds[seed_id][sl] = v[sl] - hat @ v[sl]

        # diagnostic: variance of the mean human response explained by difficulty
        y = np.asarray(human_responses, dtype=np.float64)[:, sl].mean(axis=0)
        ss_tot = ((y - y.mean()) ** 2).sum()
        ss_res = ((y - hat @ y) ** 2).sum()
        r2_blocks[b] = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    return human_resid, model_resid, {'human_r2_per_block': r2_blocks}


def difficulty_regressed_consistency(
        all_model_outputs, human_responses, similarity, match,
        task_metadata, model_splits, human_splits,
        n_pairs=100, internal_cutoff=0.1,
        aggs=('per_task', 'per_task_type', 'per_feature', 'overall'),
        feature_indices=None,
):
    """Consistency metrics after regressing out the visual-difficulty covariate.

    Same contract as ``consistency_hierarchical`` plus the covariate inputs;
    returns ``(results, samples, diagnostics)``.
    """
    human_resid, model_resid, diagnostics = residualize_on_difficulty(
        human_responses, all_model_outputs, similarity, match, n_pairs=n_pairs
    )
    results, samples = consistency_hierarchical(
        all_model_outputs=model_resid, human_responses=human_resid,
        n_pairs=n_pairs, task_metadata=task_metadata,
        model_splits=model_splits, human_splits=human_splits,
        internal_cutoff=internal_cutoff,
        aggs=aggs, feature_indices=feature_indices,
    )
    return results, samples, diagnostics


def shuffled_covariate_null(similarity, n_permutations=10, n_pairs=100, random_state=0):
    """Yield covariates with the difficulty signal destroyed but the design intact.

    Projecting out any regressor costs degrees of freedom and lowers every
    consistency score slightly, whether or not the regressor carries signal.
    Permuting ``similarity`` within each task block keeps the block structure
    and the covariate's marginal distribution while decoupling it from the
    trial, so the difference between the real and permuted runs is what the
    difficulty covariate actually explains rather than what the projection
    costs. With the two-column design this cost is small, but reporting the
    null makes the comparison unambiguous.

    Feed each yielded array through ``difficulty_regressed_consistency`` exactly
    as the real covariate, and read the difficulty effect as
    ``observed - mean(null)``.

    Args:
        similarity: (N,) covariate from ``pair_similarity``.
        n_permutations: Number of null draws to yield.
        n_pairs: Trials per task block (permutation is within-block).
        random_state: Seed for the permutation RNG.

    Yields:
        (N,) float arrays, each a within-block permutation of ``similarity``.
    """
    similarity = np.asarray(similarity, dtype=np.float64)
    assert similarity.shape[0] % n_pairs == 0
    rng = np.random.default_rng(random_state)
    for _ in range(n_permutations):
        s = similarity.copy()
        for b in range(similarity.shape[0] // n_pairs):
            sl = slice(b * n_pairs, (b + 1) * n_pairs)
            s[sl] = rng.permutation(s[sl])
        yield s
