# Analysis

Library code the scripts in [`scripts/`](../scripts/README.md) call: the model registry, the evaluation on the MTurk trials, the behavioral metrics and the DSA.

| Module | Contents |
|---|---|
| `factory.py` | model configs of the paper and the rebuilding of the trained models for evaluation |
| `transplant_configs.py` | keys and configs of the substitution models, and the arms they form |
| `eval_loop.py` | responses of models, humans and VLMs on the MTurk trials |
| `experiment_pipeline.py` | cached responses and the behavioral metrics of the paper, with the array layouts |
| `metrics/` | accuracy, consistency with humans and error consistency |
| `cross_task.py` | cross-task and cross-stimulus response correlations and their alignment (RSA) |
| `difficulty.py` | visual-difficulty covariate control |
| `summary.py`, `plot.py`, `helper.py` | metric tables, bar panels and helpers |
| `dsa/` | per-task DSA, the distances between DMD operators, and the systems compared |

## Model names

The analysis code refers to every model by a key (`analysis/factory.py`, `analysis/transplant_configs.py`).
`AB_LSTM_NOMTURK_U80` is the one-hot Abstract LSTM at its checkpoint closest to 0.80 validation accuracy; `_FINAL` marks the final checkpoint, and no suffix the epoch-0 checkpoint closest to 0.75.
`NOMTURK` means trained on the split without the MTurk images, as every model in the paper is.
Natural keys name the encoder and the core (`ResNet_CLIP_LSTM_NOMTURK`), `DIM_MATCH_LSTM_*` are the dimensionality-matched models, and `REG_XFER_*` and `DM_XFER_*` are the substitutions with one-hot and dimensionality-matched Abstract models (`ABS` or `RES`: the recipient is Abstract or Natural; `THETA`, `THETA_WOUT`, `THETA_WIN`, `UNTRAINED`: the frozen modules).
