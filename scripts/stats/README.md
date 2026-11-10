# Statistics

Commands run from the repository root; set up the environment and download the data first (see the [main README](../../README.md)).

```bash
export PYTHONHASHSEED=0
python scripts/stats/natural_vs_abstract.py   # Natural vs Abstract paired tests (Sec. 4.1)
python scripts/stats/rsa.py                   # RSA tests and the correlations between metrics (Sec. 4.1)
python scripts/stats/text_numbers.py          # validation accuracies and other numbers quoted in the text
python scripts/stats/discriminability.py      # visual discriminability control (appendix; needs HvM and features)
python scripts/stats/separability.py          # linear separability of the encoder features (needs features)
```

The separability values behind Figure 12 (`data/evaluations/vision_sgd.npy`) were computed from an earlier version of the encoder features; `separability.py` recomputes them from the feature caches into `vision_sgd_recomputed.npy`, within 0.03 of the paper's values (its docstring explains the difference).
