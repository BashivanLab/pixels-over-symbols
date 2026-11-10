# MTurk stimuli and trials

Commands run from the repository root; set up the environment and download the data first (see the [main README](../../README.md)).
The docstring of each script describes its options and the rules it follows.

## Stimuli

```bash
python scripts/mturk/build_stimuli.py OUT_DIR                                        # draw a new subset
python scripts/mturk/build_stimuli.py OUT_DIR --subset data/mturk/trials/stim_subset.csv   # export the released one
```

`build_stimuli.py` reads the full HvM image set and metadata from `data/original`.
The subset holds 100 HvM images of 16 objects, 2 from each of the 8 categories.
The released subset has this structure but was drawn with an unrecorded random state, so a new draw gives another subset.
Exporting the released subset reproduces `data/mturk/images`: the 100 images byte for byte, the gray background pixel for pixel.

## Trial lists

```bash
python scripts/mturk/build_trials.py OUT_DIR     # build the 12 trial lists from a subset
python scripts/mturk/build_trials.py --check     # check the released lists in data/mturk/trials
```

Every trial list compares the same 100 ordered stimulus pairs, drawn with match rates near 0.5 for category, object and position.
The released lists have this structure, but their pair sets, shuffles and context stimuli came from an unrecorded random state, so a rebuild gives other trials.
`--check` verifies that every answer agrees with the stimulus metadata, that all lists compare the same 100 pairs, and that each session holds 20 pairs.
