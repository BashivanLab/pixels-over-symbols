# Scripts

Commands run from the repository root; set up the environment and download the data first (see the [main README](../README.md)).

| Folder | Contents |
|---|---|
| [`train/`](train/README.md) | launchers that train every model of the paper, and the encoder feature caches |
| [`figures/`](figures/README.md) | redraw the data figures of the paper |
| [`stats/`](stats/README.md) | statistical tests and the numbers quoted in the text |
| [`dsa/`](dsa/README.md) | Dynamical Similarity Analysis of the substitution models |
| [`mturk/`](mturk/README.md) | build the MTurk stimulus subset and trial lists |

## Evaluating models

`scripts/evaluate.py` runs the checkpoints in `data/checkpoints` on the MTurk trials and caches their responses in `data/evaluations`, one group of models at a time:

```bash
python scripts/evaluate.py --list               # the groups and their models
python scripts/evaluate.py abstract natural     # evaluate two groups
```

Models whose responses are already cached are skipped, so delete their files in `data/evaluations` to recompute them.
Natural models never run their vision encoder: they read its features from `data/features`, so build those first with `scripts/train/features.sh` (this needs the HvM images).
Evaluated on a CPU, as in the paper and by default here, the released checkpoints reproduce the cached responses exactly.
The one exception is the random-initialization twins of the dimensionality-matched models, whose confidences can change in the fifth decimal with the number of CPU threads (the cache was computed with 64).
