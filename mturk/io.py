"""Paths of the MTurk stimuli.

The stimuli were shown to participants under ``images/<Variation folder>/<file name>``, with the leading
underscore of HvM file names removed; these helpers map between that form and HvM paths.
"""

from pathlib import Path

import torch


def _subpath_after(p: Path, segment: str):
    """Return the subpath (as Path) after `segment`"""
    try:
        idx = p.parts.index(segment)
    except ValueError:
        return p
    return Path(*p.parts[idx + 1:])


def mturk_localize(resource_dir, p: str):
    """An HvM path rebased under `resource_dir` as the MTurk stimuli are named (no leading underscore)."""
    relative = _subpath_after(Path(p), "HvM_with_discfade")
    relative = relative.with_name(relative.name.lstrip("_"))
    return resource_dir / relative


def localize_mturk_csv_filename(stim_df):
    """Add the stimulus names of the trial lists to a stimulus dataframe.

    ``re_filename``: ``filename`` without the underscore after each '/'; ``local_fp``: its part after
    'HvM_with_discfade/', i.e. ``Variation.../file.png``.
    """
    if 're_filename' not in stim_df.columns:
        if isinstance(stim_df['filename'].iloc[0], Path):
            stim_df['re_filename'] = stim_df['filename'].apply(
                lambda p: str(p)
            ).str.replace(r'(?<=/)_', '', regex=True)
        else:
            stim_df['re_filename'] = stim_df['filename'].str.replace(r'(?<=/)_', '', regex=True)

    stim_df['local_fp'] = stim_df['re_filename'].apply(
        lambda p: _subpath_after(Path(p), segment='HvM_with_discfade')
    )
    return stim_df


def _load_image_with_fallback(
        img_path: Path,
        img_loader,
        stim_dir: Path
) -> torch.Tensor:
    """Load an MTurk stimulus from `stim_dir`, under its MTurk name or with the underscore prefix of HvM.

    Raises:
        ValueError: If image cannot be loaded from either path
    """
    subpath = _subpath_after(img_path, 'images')

    try:
        return img_loader.read_img(stim_dir / subpath)
    except (ValueError, FileNotFoundError, OSError) as e:
        parts = str(subpath).split('/')
        parts[-1] = '_' + parts[-1]
        fallback_path = '/'.join(parts)

        try:
            return img_loader.read_img(stim_dir / fallback_path)
        except Exception:
            raise ValueError(
                f"Failed to load image from both paths:\n"
                f"  1. {stim_dir / subpath}\n"
                f"  2. {stim_dir / fallback_path}"
            ) from e
