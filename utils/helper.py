"""Miscellaneous helpers shared across training, tasks, and analysis.

Small utilities: checkpoint repair for ``torch.compile``-wrapped state dicts,
reproducibility seeding, per-trial stimulus lookup, CUDA/model printouts, and
feature-cache lookup.
"""

import os
import random
from pathlib import Path

import numpy as np
import torch

from utils.paths import FEATURE_DIR


def is_compile_supported():
    """Return True if a CUDA GPU with compute capability >= 7 (Volta+) is present for ``torch.compile``."""
    if torch.cuda.is_available():
        try:
            props = torch.cuda.get_device_properties(0)
            return props.major >= 7
        except Exception as e:
            print(f"[Warning] Could not get GPU properties: {e}")
    return False


def remove_prefix(text, prefix):
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def load_repair_checkpoint(path, device='cpu'):
    """Load a state dict, stripping the ``_orig_mod.`` prefix ``torch.compile`` adds to its keys."""
    ckpt = torch.load(
        path,
        map_location=torch.device(device)
    )
    in_state_dict = ckpt
    pairings = [
        (src_key, remove_prefix(src_key, "_orig_mod."))
        for src_key in in_state_dict.keys()
    ]
    if all(src_key == dest_key for src_key, dest_key in pairings):
        return ckpt
    out_state_dict = {}
    for src_key, dest_key in pairings:
        out_state_dict[dest_key] = in_state_dict[src_key]
    ckpt = out_state_dict
    return ckpt


def get_one_hot(index: int, total: int = 43):
    """Return a length-``total`` one-hot vector for the 1-based ``index`` (used for task ids)."""
    result = np.zeros(total)
    result[index - 1] = 1
    return result


def get_per_trial_info(trial_info_dict, hvm_loader_df, mapper):
    """
    get the trial information
    """
    meta_row_lists = {
        "row_idx": list(),
        "category": list(),
        "identity": list(),
        "position": list(),
        "view": list(),
        "size": list(),
        "var": list(),
    }
    for frame_t, t in trial_info_dict.items():
        if t is not None:
            cat, obj, pos, view, size = t
            to_int = lambda x: x.item() if torch.is_tensor(x) else int(x)
            to_float = lambda x: x.item() if torch.is_tensor(x) else float(x)

            cat, obj, pos = map(to_int, (cat, obj, pos))
            size = to_float(size)
            view = tuple(to_float(e) for e in view)

            row_idx = mapper.find(t)
            var = int(hvm_loader_df.iloc[row_idx]['var'])

            meta_row_lists['row_idx'].append(row_idx)
            meta_row_lists['category'].append(cat)
            meta_row_lists['identity'].append(obj)
            meta_row_lists['position'].append(pos)
            meta_row_lists['view'].append(view)
            meta_row_lists['size'].append(size)
            meta_row_lists['var'].append(var)
        else:
            for v in meta_row_lists.values():
                v.append(-1)
    return meta_row_lists


def seed_everything(seed: int = 2025):
    """Seed Python, NumPy, and torch (CPU + CUDA) RNGs for reproducible runs."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seeded everything with seed {seed}")
    return


def worker_init_fn(worker_id):
    """DataLoader worker initializer that reseeds NumPy/Python per worker for varied sampling."""
    seed = torch.initial_seed() % (2 ** 32 - 1)
    np.random.seed(seed)
    random.seed(seed)


def print_model_info(model, print_model=True):
    if print_model:
        print("\nModel Structure:")
        print('\t', model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print('total params: ', pytorch_total_params)
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('trainable params: ', pytorch_total_params)
    return


def print_cuda_info():
    print("=== PyTorch CUDA Info ===")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        current_device = torch.cuda.current_device()
        print(f"torch.cuda.current_device(): {current_device}")
        print(f"torch.cuda.get_device_name({current_device}): {torch.cuda.get_device_name(current_device)}")
        print(f"GPU memory allocated: {torch.cuda.memory_allocated() / 1024 ** 3:.4f} GB")
        print(f"GPU memory reserved: {torch.cuda.memory_reserved() / 1024 ** 3:.4f} GB")
    else:
        print("CUDA not available, skipping detailed GPU info.")
    return


def find_emb_cache(cnn_type, emb_cache_dir):
    """Return the four train/val feature+blank cache paths for ``cnn_type`` if all exist, else ``[]``."""
    if emb_cache_dir is None:
        emb_cache_dir = FEATURE_DIR
    else:
        # configs and CLI flags supply this as a str; the joins below need a Path
        emb_cache_dir = Path(emb_cache_dir)
    cached_train_emb_path = emb_cache_dir / f'{cnn_type}_train_features.pt'
    cached_val_emb_path = emb_cache_dir / f'{cnn_type}_val_features.pt'
    blank_train_emb_path = emb_cache_dir / f'{cnn_type}_train_blank.pt'
    blank_val_emb_path = emb_cache_dir / f'{cnn_type}_val_blank.pt'

    paths = [cached_train_emb_path, cached_val_emb_path, blank_train_emb_path, blank_val_emb_path]
    if all(p.is_file() for p in paths):
        find_cached = True
    else:
        print("[Natural] --- CNN Cache not found. Running with raw images (Backbone active). ---")
        find_cached = False
        paths = list()

    if find_cached:
        # the directory belongs in the log: caches are keyed only by backbone
        # name, so the same filename under a different directory is a different
        # encoder and nothing downstream can tell them apart
        print(f"[Natural] ---  Using Cached Features for {cnn_type} from {emb_cache_dir} ---")
    return paths
