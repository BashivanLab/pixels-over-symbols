"""Layout of the MTurk trial lists and participant files of every task.

:class:`TaskConfig` names a task's trial list and its stimulus (``stim<k>``) and answer (``act<k>``)
columns; ``TASK_CONFIGS`` covers the 43 tasks, of which the 12 with a trial list in
data/mturk/trials were run on MTurk.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict

import pandas as pd

from utils.paths import TRIAL_DIR

FEATURE_TO_TASK_NAME = {
    "pos_1b": "position",
    "obj": "identity",
    "cat_1b": "category",
}

FEATURE_TO_TRIAL_TOKEN = {
    "pos_1b": "pos",
    "obj": "obj",
    "cat_1b": "category",
}


@dataclass(frozen=True)
class TaskConfig:
    task_name: str
    task_id: int

    stim_cols: List[str]
    act_cols: List[str]
    act_indices: List[int]  # 1-based positions of the graded stimuli

    features: List[str]

    task_trial_csv: Path
    task_meta_csv: Path

    num_sessions: int = 5
    trials_per_session: int = 20
    frames_per_trial: int = 0

    resp_col: str = "resp.keys"
    rt_col: str = "resp.rt"
    trial_start_col: str = "TrialIntro.started"
    session_col: str = "session"


def stim_cols(n_frames: int) -> List[str]:
    return [f"stim{i}" for i in range(1, n_frames + 1)]


def act_cols_from_indices(indices: List[int]) -> List[str]:
    return [f"act{i}" for i in indices]


def make_nback_task(
        *,
        n_back: int,
        feature: str,
        task_id: int,
        n_frames: int,
) -> TaskConfig:
    assert feature in FEATURE_TO_TASK_NAME
    assert n_frames > n_back

    # the first n_back stimuli have nothing n_back steps back to compare with
    act_indices = list(range(n_back + 1, n_frames + 1))

    name_token = FEATURE_TO_TASK_NAME[feature]
    csv_token = FEATURE_TO_TRIAL_TOKEN[feature]

    return TaskConfig(
        task_name=f"{n_back}back_{name_token}",
        task_id=task_id,

        stim_cols=stim_cols(n_frames),
        act_cols=act_cols_from_indices(act_indices),
        act_indices=act_indices,

        features=[feature],

        task_trial_csv=TRIAL_DIR / f"{n_back}back_{csv_token}_trials.csv",
        task_meta_csv=TRIAL_DIR / "stim_subset.csv",

        frames_per_trial=n_frames,
    )


def make_interdms_task(
        *,
        pattern: str,  # "AABB" or "ABBA"
        features: List[str],  # e.g. ["pos_1b", "obj"]
        task_id: int,
) -> TaskConfig:
    name_tokens = [FEATURE_TO_TASK_NAME[f] for f in features]
    csv_tokens = [FEATURE_TO_TRIAL_TOKEN[f] for f in features]

    return TaskConfig(
        task_name=f"interdms_{pattern}_{'_'.join(name_tokens)}",
        task_id=task_id,

        stim_cols=stim_cols(4),
        act_cols=["act3", "act4"],
        act_indices=[3, 4],

        features=features,

        task_trial_csv=TRIAL_DIR / f"interdms_{pattern}_{'_'.join(csv_tokens)}_trials.csv",
        task_meta_csv=TRIAL_DIR / "stim_subset.csv",

        frames_per_trial=4,
    )


def make_dms_task(
        *,
        feature: str,
        task_id: int,
) -> TaskConfig:
    assert feature in FEATURE_TO_TASK_NAME

    name_token = FEATURE_TO_TASK_NAME[feature]
    csv_token = FEATURE_TO_TRIAL_TOKEN[feature]

    return TaskConfig(
        task_name=f"dms_{name_token}",
        task_id=task_id,

        stim_cols=["stim1", "stim2"],
        act_cols=["act2"],
        act_indices=[2],

        features=[feature],

        task_trial_csv=TRIAL_DIR / f"dms_{csv_token}_trials.csv",
        task_meta_csv=TRIAL_DIR / "stim_subset.csv",

        frames_per_trial=2,
    )


def make_ctxdms_task(
        *,
        features: List[str],  # context feature first, e.g. ["pos_1b", "cat_1b", "obj"]
        task_id: int,
) -> TaskConfig:
    name_tokens = [FEATURE_TO_TASK_NAME[f] for f in features]
    csv_tokens = [FEATURE_TO_TRIAL_TOKEN[f] for f in features]

    return TaskConfig(
        task_name=f"ctxdms_{'_'.join(name_tokens)}",
        task_id=task_id,

        stim_cols=stim_cols(3),
        act_cols=["act3"],
        act_indices=[3],

        features=features,

        task_trial_csv=TRIAL_DIR / f"ctxdms_{'_'.join(csv_tokens)}_trials.csv",
        task_meta_csv=TRIAL_DIR / "stim_subset.csv",

        frames_per_trial=3,
    )


TASK_CONFIGS: Dict[int, TaskConfig] = {}

# ---- 1-back ----
TASK_CONFIGS[1] = make_nback_task(n_back=1, feature="pos_1b", task_id=1, n_frames=6)
TASK_CONFIGS[2] = make_nback_task(n_back=1, feature="obj", task_id=2, n_frames=6)
TASK_CONFIGS[3] = make_nback_task(n_back=1, feature="cat_1b", task_id=3, n_frames=6)

# ---- 2-back ----
TASK_CONFIGS[4] = make_nback_task(n_back=2, feature="pos_1b", task_id=4, n_frames=6)
TASK_CONFIGS[5] = make_nback_task(n_back=2, feature="obj", task_id=5, n_frames=6)
TASK_CONFIGS[6] = make_nback_task(n_back=2, feature="cat_1b", task_id=6, n_frames=6)

# ---- 3-back ----
TASK_CONFIGS[7] = make_nback_task(n_back=3, feature="pos_1b", task_id=7, n_frames=6)
TASK_CONFIGS[8] = make_nback_task(n_back=3, feature="obj", task_id=8, n_frames=6)
TASK_CONFIGS[9] = make_nback_task(n_back=3, feature="cat_1b", task_id=9, n_frames=6)

# ---- interDMS AABB ----
TASK_CONFIGS[10] = make_interdms_task(pattern="AABB", features=["pos_1b", "pos_1b"], task_id=10)
TASK_CONFIGS[11] = make_interdms_task(pattern="AABB", features=["pos_1b", "obj"], task_id=11)
TASK_CONFIGS[12] = make_interdms_task(pattern="AABB", features=["pos_1b", "cat_1b"], task_id=12)
TASK_CONFIGS[13] = make_interdms_task(pattern="AABB", features=["obj", "pos_1b"], task_id=13)
TASK_CONFIGS[14] = make_interdms_task(pattern="AABB", features=["obj", "obj"], task_id=14)
TASK_CONFIGS[15] = make_interdms_task(pattern="AABB", features=["obj", "cat_1b"], task_id=15)
TASK_CONFIGS[16] = make_interdms_task(pattern="AABB", features=["cat_1b", "pos_1b"], task_id=16)
TASK_CONFIGS[17] = make_interdms_task(pattern="AABB", features=["cat_1b", "obj"], task_id=17)
TASK_CONFIGS[18] = make_interdms_task(pattern="AABB", features=["cat_1b", "cat_1b"], task_id=18)

# ---- interDMS ABAB ----
TASK_CONFIGS[19] = make_interdms_task(pattern="ABAB", features=["pos_1b", "pos_1b"], task_id=19)
TASK_CONFIGS[20] = make_interdms_task(pattern="ABAB", features=["pos_1b", "obj"], task_id=20)
TASK_CONFIGS[21] = make_interdms_task(pattern="ABAB", features=["pos_1b", "cat_1b"], task_id=21)
TASK_CONFIGS[22] = make_interdms_task(pattern="ABAB", features=["obj", "pos_1b"], task_id=22)
TASK_CONFIGS[23] = make_interdms_task(pattern="ABAB", features=["obj", "obj"], task_id=23)
TASK_CONFIGS[24] = make_interdms_task(pattern="ABAB", features=["obj", "cat_1b"], task_id=24)
TASK_CONFIGS[25] = make_interdms_task(pattern="ABAB", features=["cat_1b", "pos_1b"], task_id=25)
TASK_CONFIGS[26] = make_interdms_task(pattern="ABAB", features=["cat_1b", "obj"], task_id=26)
TASK_CONFIGS[27] = make_interdms_task(pattern="ABAB", features=["cat_1b", "cat_1b"], task_id=27)

# ---- interDMS ABBA ----
TASK_CONFIGS[28] = make_interdms_task(pattern="ABBA", features=["pos_1b", "pos_1b"], task_id=28)
TASK_CONFIGS[29] = make_interdms_task(pattern="ABBA", features=["pos_1b", "obj"], task_id=29)
TASK_CONFIGS[30] = make_interdms_task(pattern="ABBA", features=["pos_1b", "cat_1b"], task_id=30)
TASK_CONFIGS[31] = make_interdms_task(pattern="ABBA", features=["obj", "pos_1b"], task_id=31)
TASK_CONFIGS[32] = make_interdms_task(pattern="ABBA", features=["obj", "obj"], task_id=32)
TASK_CONFIGS[33] = make_interdms_task(pattern="ABBA", features=["obj", "cat_1b"], task_id=33)
TASK_CONFIGS[34] = make_interdms_task(pattern="ABBA", features=["cat_1b", "pos_1b"], task_id=34)
TASK_CONFIGS[35] = make_interdms_task(pattern="ABBA", features=["cat_1b", "obj"], task_id=35)
TASK_CONFIGS[36] = make_interdms_task(pattern="ABBA", features=["cat_1b", "cat_1b"], task_id=36)

# ---- ctxDMS ----
TASK_CONFIGS[37] = make_ctxdms_task(
    features=["pos_1b", "cat_1b", "obj"],
    task_id=37,
)

TASK_CONFIGS[38] = make_ctxdms_task(
    features=["pos_1b", "obj", "cat_1b"],
    task_id=38,
)

TASK_CONFIGS[39] = make_ctxdms_task(
    features=["obj", "pos_1b", "cat_1b"],
    task_id=39,
)

TASK_CONFIGS[40] = make_ctxdms_task(
    features=["cat_1b", "obj", "pos_1b"],
    task_id=40,
)

# ---- DMS ----
TASK_CONFIGS[41] = make_dms_task(feature="pos_1b", task_id=41)
TASK_CONFIGS[42] = make_dms_task(feature="obj", task_id=42)
TASK_CONFIGS[43] = make_dms_task(feature="cat_1b", task_id=43)

logger = logging.getLogger(__name__)


def validate_task_configs(
        task_configs: Dict[int, TaskConfig] = TASK_CONFIGS,
        *,
        strict: bool = False,
) -> Dict[int, TaskConfig]:
    """The configs whose trial list and stimulus subset exist and have the expected columns.

    A config that fails is left out, or raises an AssertionError with ``strict``.
    """

    def fail(msg):
        if strict:
            raise AssertionError(msg)
        else:
            logger.debug(msg)

    checked_configs = dict()

    for task_id, cfg in task_configs.items():
        if not cfg.task_trial_csv.exists():
            fail(f"[{cfg.task_name}] Trial CSV missing: {cfg.task_trial_csv}")
            continue

        if not cfg.task_meta_csv.exists():
            fail(f"[{cfg.task_name}] Stim CSV missing: {cfg.task_meta_csv}")
            continue

        trial_df = pd.read_csv(cfg.task_trial_csv)
        stim_df = pd.read_csv(cfg.task_meta_csv)

        for col in cfg.stim_cols:
            if col not in trial_df.columns:
                fail(f"[{cfg.task_name}] Missing stim column in trial CSV: {col}")

        for col in cfg.act_cols:
            if col not in trial_df.columns:
                fail(f"[{cfg.task_name}] Missing act column in trial CSV: {col}")

        if len(cfg.stim_cols) != cfg.frames_per_trial:
            fail(
                f"[{cfg.task_name}] frames_per_trial mismatch: "
                f"{cfg.frames_per_trial} vs {len(cfg.stim_cols)}"
            )

        if "filename" not in stim_df.columns:
            fail(f"[{cfg.task_name}] Stim CSV missing filename")

        logger.debug(f"[OK] {task_id:02d} {cfg.task_name}")
        checked_configs[task_id] = cfg

    return checked_configs
