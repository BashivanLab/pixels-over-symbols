"""Shared constants for the 43-task battery.

Defines the task-name to task-index mapping, task-type groupings, model
input/output sizes, pretrained normalization statistics, and the list of VLM
models, so task ids and dimensions stay consistent across trial generation,
training, and analysis.
"""

TASK_NAME_TASK_INDEX = {
    '1back_position': 1,
    '1back_identity': 2,
    '1back_category': 3,
    '2back_position': 4,
    '2back_identity': 5,
    '2back_category': 6,
    '3back_position': 7,
    '3back_identity': 8,
    '3back_category': 9,

    'interdms_AABB_position_position': 10,
    'interdms_AABB_position_identity': 11,
    'interdms_AABB_position_category': 12,
    'interdms_AABB_identity_position': 13,
    'interdms_AABB_identity_identity': 14,
    'interdms_AABB_identity_category': 15,
    'interdms_AABB_category_position': 16,
    'interdms_AABB_category_identity': 17,
    'interdms_AABB_category_category': 18,

    'interdms_ABAB_position_position': 19,
    'interdms_ABAB_position_identity': 20,
    'interdms_ABAB_position_category': 21,
    'interdms_ABAB_identity_position': 22,
    'interdms_ABAB_identity_identity': 23,
    'interdms_ABAB_identity_category': 24,
    'interdms_ABAB_category_position': 25,
    'interdms_ABAB_category_identity': 26,
    'interdms_ABAB_category_category': 27,

    'interdms_ABBA_position_position': 28,
    'interdms_ABBA_position_identity': 29,
    'interdms_ABBA_position_category': 30,
    'interdms_ABBA_identity_position': 31,
    'interdms_ABBA_identity_identity': 32,
    'interdms_ABBA_identity_category': 33,
    'interdms_ABBA_category_position': 34,
    'interdms_ABBA_category_identity': 35,
    'interdms_ABBA_category_category': 36,

    'ctxdms_position_category_identity': 37,
    'ctxdms_position_identity_category': 38,
    'ctxdms_identity_position_category': 39,
    'ctxdms_category_identity_position': 40,

    'dms_position': 41,
    'dms_identity': 42,
    'dms_category': 43
}

TASK_TYPE_DICT = dict()
for i in range(1, 44):
    if i < 10:
        TASK_TYPE_DICT[i] = 'NBack'
    elif i < 37:
        TASK_TYPE_DICT[i] = 'InterDMS'
    elif i < 41:
        TASK_TYPE_DICT[i] = 'CtxDMS'
    else:
        TASK_TYPE_DICT[i] = 'DMS'

TASK_METADATA = {
    task_idx: {
        'name': task_name,
        'type': TASK_TYPE_DICT[task_idx]
    }
    for task_name, task_idx in TASK_NAME_TASK_INDEX.items()
}

# Training constants
INPUT_SIZE = 8 + 64 + 9 + 3 + 1 + 43  # 8 categories + 64 objects + 9 positions + 3 view + 1 size + 43 tasks = 128
OUTPUT_SIZE = 2

PRETRAINED_STATS = {
    'resnet': ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    'clip_resnet': ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
}

VLM_MODELS = [
    "nemotron-nano-12b-v2-vl",
    "nemotron-nano-12b-v2-vl-_no_think",
    "qwen3-vl-30b-a3b-thinking",
    "qwen3-vl-235b-a22b-thinking",
    "gemini-3.1-pro-preview",
]

