"""Where the code looks for data, checkpoints and cached results.

Everything sits under data/ at the repository root; symlink data/ (or any folder in it) to
wherever the files actually live.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / 'data'

# HvM images and metadata (Majaj et al. 2015), not redistributed with this code
HVM_DIR = DATA_DIR / 'original'
# MTurk trial lists, the 100 MTurk stimuli, and the human and VLM responses to those trials
MTURK_DIR = DATA_DIR / 'mturk'
TRIAL_DIR = MTURK_DIR / 'trials'
HUMAN_DIR = MTURK_DIR / 'human'
VLM_DIR = MTURK_DIR / 'vlm'
# frozen-backbone features written by cache_image_embed.py
FEATURE_DIR = DATA_DIR / 'features'
# trained models, one folder per run
CHECKPOINT_DIR = DATA_DIR / 'checkpoints'
# cached model, human and VLM responses on the MTurk trials, DSA results
EVAL_DIR = DATA_DIR / 'evaluations'
FIGURE_DIR = ROOT / 'figures'
