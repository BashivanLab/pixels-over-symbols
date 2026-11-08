"""Linear separability of category, identity and position in the frozen vision encoders (fig:metric_alignment).

For each of the 10 encoders of the vision-encoder sweep, a linear SVM trained by SGD (scikit-learn's
SGDClassifier, hinge loss, one-vs-rest) predicts the category (8 classes), identity (64) or position
(9 grid cells) of every HvM image from the encoder's flattened cached features, with 5-fold stratified
cross-validation. The out-of-fold accuracy is the linear separability of that feature in that encoder.

usage: python scripts/stats/separability.py
Reads the feature caches in FEATURE_DIR (cache_image_embed.py) and the HvM metadata, and writes
EVAL_DIR/vision_sgd_recomputed.npy: {encoder: {'cat' | 'id' | 'pos': {'overall_accuracy',
'per_var_accuracy', 'report', 'y_oof'}}}, y_oof in the order of the HvM metadata. About 1 h 40 min with
8 threads and up to 23 GB of memory (EVA02-E).

The figures read EVAL_DIR/vision_sgd.npy, the values of the paper, which this script does not overwrite.
They came from the same classifier and cross-validation, but on features
computed in float32 from each encoder run on the images with the CLIP normalization (EVA02-E: from
an earlier version of its cache) and with the images in another order, which changes the folds.
This script reads the float16 caches the models were trained on; its accuracies differ from the
paper's by up to 0.03 (identity, ResNet-50 IN1k), and by less than 0.01 for 23 of the 30
encoder-feature pairs.
"""
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from utils.paths import EVAL_DIR, FEATURE_DIR, HVM_DIR  # noqa: E402
from utils.stim_io import HvMMetaData  # noqa: E402

ENCODERS = [
    'resnet50.a1_in1k', 'resnet50_clip.openai',
    'vit_base_patch16_224.orig_in21k', 'vit_base_patch16_clip_224.laion2b_ft_in1k',
    'vit_base_patch16_224.dino', 'vit_base_patch16_dinov3.lvd1689m',
    'eva02_base_patch14_224.mim_in22k', 'eva02_base_patch16_clip_224.merged2b',
    'eva02_large_patch14_clip_224.merged2b', 'eva02_enormous_patch14_clip_224.laion2b_plus',
]
SEED = 2025


def cross_val_sgd(X, y, var, n_splits=5, random_state=SEED):
    """Out-of-fold predictions of a linear SVM (SGD, hinge loss) with stratified k-fold cross-validation.

    Returns the overall accuracy, the accuracy per variation level, the classification report and the
    out-of-fold predictions.
    """
    folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    y_oof = np.empty_like(y)
    for train, test in folds.split(X, y):
        # n_jobs only spreads the one-vs-rest problems over threads; each draws its seed up front
        model = SGDClassifier(loss='hinge', max_iter=1000, tol=1e-3, random_state=random_state, n_jobs=8)
        y_oof[test] = model.fit(X[train], y[train]).predict(X[test])
    return {
        'overall_accuracy': float(accuracy_score(y, y_oof)),
        'per_var_accuracy': {int(v): float(accuracy_score(y[var == v], y_oof[var == v])) for v in np.unique(var)},
        'report': classification_report(y, y_oof, digits=4),
        'y_oof': y_oof,
    }


def main():
    df = HvMMetaData(HVM_DIR).df
    cat, ident, pos = (df[c].to_numpy(dtype=int) - 1 for c in ('cat_1b', 'id_1b', 'pos_1b'))
    labels = {'cat': cat, 'id': cat * 8 + ident, 'pos': pos}
    var = df['var'].to_numpy(dtype=int)

    results = dict()
    for encoder in ENCODERS:
        t0 = time.time()
        # the train cache holds every HvM image, in metadata order
        cache = torch.load(FEATURE_DIR / f'{encoder}_train_features.pt')
        assert len(cache) == len(df), f'{encoder}: {len(cache)} cached images, {len(df)} in the metadata'
        X = cache.float().reshape(len(cache), -1).numpy()
        del cache
        results[encoder] = {name: cross_val_sgd(X, y, var) for name, y in labels.items()}
        print(f'{encoder}: ' + ', '.join(f"{name} {r['overall_accuracy']:.4f}" for name, r in results[encoder].items())
              + f' ({time.time() - t0:.0f}s)', flush=True)
    np.save(EVAL_DIR / 'vision_sgd_recomputed.npy', results, allow_pickle=True)
    print('saved', EVAL_DIR / 'vision_sgd_recomputed.npy')


if __name__ == '__main__':
    main()
