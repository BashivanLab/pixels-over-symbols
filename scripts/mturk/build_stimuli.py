"""Draw the MTurk stimulus subset and export its images as they were shown to participants.

usage: python scripts/mturk/build_stimuli.py OUT_DIR [--seed SEED] [--subset CSV]

The subset holds 100 HvM images of 16 objects, 2 from each of the 8 categories, each object at the centre
of the 3 x 3 position grid and at random other positions, 7 images for the first 4 objects and 6 for the
others (mturk.stimuli.draw_stimulus_subset). It is written to OUT_DIR/stim_subset.csv, with file names
relative to the HvM folder (data/original); --subset exports the images of an existing subset instead,
e.g. data/mturk/trials/stim_subset.csv. The images are copied to OUT_DIR/images/<Variation
folder>/<file name> without the leading underscore of the HvM names, next to the uniform gray
images/gray_background.png.

The released subset has this structure but was drawn with an unrecorded random state, so a new draw gives
another subset. Exporting the released subset reproduces data/mturk/images: the 100 images byte for byte,
the gray background pixel for pixel.
"""
import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from mturk.io import mturk_localize  # noqa: E402
from mturk.stimuli import draw_stimulus_subset  # noqa: E402
from utils.helper import seed_everything  # noqa: E402
from utils.paths import HVM_DIR  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('out_dir', type=Path)
    parser.add_argument('--seed', type=int, default=2025, help='seed of the subset draw')
    parser.add_argument('--subset', type=Path, help='export the images of this stim_subset.csv instead of drawing one')
    parser.add_argument('--hvm-dir', type=Path, default=HVM_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.subset is None:
        seed_everything(args.seed)
        subset = draw_stimulus_subset(args.hvm_dir, n_objs=16, n_stimuli=100)
        subset['filename'] = [Path(f).relative_to(args.hvm_dir).as_posix() for f in subset['filename']]
        subset.to_csv(args.out_dir / 'stim_subset.csv')
        print(f"{len(subset)} stimuli of {len(subset[['cat_1b', 'id_1b']].drop_duplicates())} objects -> "
              f"{args.out_dir / 'stim_subset.csv'}")
    else:
        subset = pd.read_csv(args.subset)

    for f in subset['filename']:
        dst = args.out_dir / mturk_localize(Path('images'), f)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.hvm_dir / f, dst)
    Image.new('RGB', (256, 256), (128, 128, 128)).save(args.out_dir / 'images' / 'gray_background.png')
    print(f"{len(subset)} images -> {args.out_dir / 'images'}")


if __name__ == '__main__':
    main()
