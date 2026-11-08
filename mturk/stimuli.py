"""The MTurk stimulus subset: 100 HvM images of 16 objects, 2 per category.

The subset was drawn once (:func:`draw_stimulus_subset`, scripts/mturk/build_stimuli.py) and is stored in
data/mturk/trials/stim_subset.csv; :func:`sample_df` loads it.
"""

import random
from pathlib import Path

import pandas as pd

from utils.stim_io import HvMImageLoader, HvMMetaData, localize_csv_filename


def pick_objs(df, n_objs):
    """``n_objs`` (category, object) pairs, the same number from every category, the remainder from random categories."""
    catids = list()
    cats = sorted(df['cat_1b'].unique())

    obj_per_cat = n_objs // len(cats)
    remainder = n_objs % len(cats)

    for cat in cats:
        objs_in_cat = sorted(df.loc[df['cat_1b'] == cat, 'id_1b'].unique())
        chosen = random.sample(objs_in_cat, obj_per_cat)
        for obj in chosen:
            catids.append((cat, obj))

    if remainder > 0:
        extra_cats = random.sample(cats, k=remainder)
        for cat in extra_cats:
            objs_in_cat = sorted(df.loc[df['cat_1b'] == cat, 'id_1b'].unique())
            already = {obj for (c, obj) in catids if c == cat}
            available = list(set(objs_in_cat) - already)
            if available:
                catids.append((cat, random.choice(available)))
    return catids


def pick_locations(df, catid_to_positions, chosen_objs, n_stimuli, loc_c=5):
    """One image per (object, grid position), ``n_stimuli`` in total, spread evenly over the objects.

    Every object gets position `loc_c` (5, the centre of the 3 x 3 grid) if it was shown there, plus
    random other positions.

    Args:
        df (pd.DataFrame): The full metadata DataFrame.
        catid_to_positions (dict): Map from (cat, obj) to list of available positions.
        chosen_objs (list): List of (cat, obj) tuples selected by pick_objs.
        n_stimuli (int): The target total number of stimulus rows to return.
        loc_c (int): The location to prioritize for every object.

    Returns:
        list: A list of sampled rows (pd.Series) forming the subset DataFrame.
    """
    rows = list()
    n_objects = len(chosen_objs)

    if n_objects == 0:
        return rows

    base_locs_per_obj = n_stimuli // n_objects
    remainder_stimuli = n_stimuli % n_objects

    for i, catid in enumerate(chosen_objs):
        cat, obj = catid
        available_positions = sorted(catid_to_positions[catid])

        locs_to_pick = base_locs_per_obj
        if i < remainder_stimuli:
            locs_to_pick += 1

        chosen_pos = []
        if loc_c in available_positions:
            chosen_pos.append(loc_c)
            remaining_available = [pos for pos in available_positions if pos != loc_c]
            locs_to_pick -= 1
        else:
            remaining_available = available_positions

        actual_remaining_to_sample = min(max(0, locs_to_pick), len(remaining_available))
        if actual_remaining_to_sample > 0:
            extra_pos = random.sample(remaining_available, k=actual_remaining_to_sample)
            chosen_pos.extend(extra_pos)

        for pos in chosen_pos:
            subset = df[
                (df["cat_1b"] == cat) &
                (df["id_1b"] == obj) &
                (df["pos_1b"] == pos)
                ]
            rows.append(subset.sample(1))

    return rows


def draw_stimulus_subset(hvm_dir, n_objs, n_stimuli, grid_size=3):
    """A new stimulus subset of ``n_stimuli`` images of ``n_objs`` objects (:func:`pick_objs`, :func:`pick_locations`).

    Draws from Python's and NumPy's global random state, so seed both first.
    """
    img_loader = HvMImageLoader(root_dir=hvm_dir, metadata=HvMMetaData(hvm_dir), preload_images=False)
    img_loader.prepare_for_tasks(grid_size)
    df = img_loader.df
    chosen_objs = pick_objs(df, n_objs)
    rows = pick_locations(df, img_loader._task_cache.catid_to_positions, chosen_objs, n_stimuli)
    return pd.concat(rows, ignore_index=True)


def sample_df(
        hvm_dir,
        n_objs,
        n_stimuli,
        grid_size,
        df_path,
        pretrained_stats=None,
):
    """HvM metadata and an image loader restricted to the stimulus subset in `df_path`.

    ``n_objs`` and ``n_stimuli`` are the size the subset must have.
    """
    if not Path(df_path).is_file():
        raise FileNotFoundError(f'No stimulus subset at {df_path}')
    meta = HvMMetaData(hvm_dir)
    img_loader = HvMImageLoader(
        root_dir=hvm_dir,
        metadata=meta,
        preload_images=False,
        pretrained_stats=pretrained_stats
    )
    df_subset = pd.read_csv(df_path)
    # file names are stored relative to the HvM folder; use the local paths HvMMetaData indexes by
    df_subset['filename'] = df_subset['filename'].map(lambda p: str(localize_csv_filename(meta.stim_dir, p)))
    n_found = df_subset[['cat_1b', 'id_1b']].drop_duplicates().shape[0]
    assert (len(df_subset), n_found) == (n_stimuli, n_objs), \
        f'{df_path}: {len(df_subset)} stimuli of {n_found} objects, expected {n_stimuli} of {n_objs}'
    img_loader.df = df_subset
    img_loader.prepare_for_tasks(grid_size)
    return img_loader, meta
