"""Multi-task composition over all DMS/inter-DMS/context-DMS/n-back variants.

``TASK_DATALOADERS`` registers every task name, grouped by task type, with its
dataset class and kwargs (``ALL_TASKS`` is the flat version), and
``TASK_IDX_TASK_TYPE`` maps the 43 task ids to their dataset classes.
:class:`MultiTaskDataset` samples across all 43 tasks so a single model trains on
the full task battery.
"""

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from task.base import HvMTaskDataset
from task.dms import DMSDataset, CtxDMSDataset, InterDMSDataset
from task.nback import NBackDataset

TASK_DATALOADERS = {
    'dms': {
        'dms_category': (DMSDataset, {"feature": "category"}),
        'dms_identity': (DMSDataset, {"feature": "identity"}),
        'dms_position': (DMSDataset, {"feature": "position"}),
    },
    'nback': {
        '1back_category': (NBackDataset, {"feature": "category", "nback_n": 1}),
        '2back_category': (NBackDataset, {"feature": "category", "nback_n": 2}),
        '3back_category': (NBackDataset, {"feature": "category", "nback_n": 3}),

        '1back_position': (NBackDataset, {"feature": "position", "nback_n": 1}),
        '2back_position': (NBackDataset, {"feature": "position", "nback_n": 2}),
        '3back_position': (NBackDataset, {"feature": "position", "nback_n": 3}),

        '1back_identity': (NBackDataset, {"feature": "identity", "nback_n": 1}),
        '2back_identity': (NBackDataset, {"feature": "identity", "nback_n": 2}),
        '3back_identity': (NBackDataset, {"feature": "identity", "nback_n": 3}),
    },
    'interdms': {
        'interdms_AABB_category_category': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'category', 'feature_2': 'category'}),
        'interdms_AABB_category_identity': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'category', 'feature_2': 'identity'}),
        'interdms_AABB_category_position': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'category', 'feature_2': 'position'}),
        'interdms_AABB_identity_category': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'identity', 'feature_2': 'category'}),
        'interdms_AABB_identity_identity': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'identity', 'feature_2': 'identity'}),
        'interdms_AABB_identity_position': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'identity', 'feature_2': 'position'}),
        'interdms_AABB_position_category': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'position', 'feature_2': 'category'}),
        'interdms_AABB_position_identity': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'position', 'feature_2': 'identity'}),
        'interdms_AABB_position_position': (
            InterDMSDataset, {'pattern': 'AABB', 'feature_1': 'position', 'feature_2': 'position'}),
        'interdms_ABAB_category_category': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'category', 'feature_2': 'category'}),
        'interdms_ABAB_category_identity': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'category', 'feature_2': 'identity'}),
        'interdms_ABAB_category_position': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'category', 'feature_2': 'position'}),
        'interdms_ABAB_identity_category': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'identity', 'feature_2': 'category'}),
        'interdms_ABAB_identity_identity': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'identity', 'feature_2': 'identity'}),
        'interdms_ABAB_identity_position': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'identity', 'feature_2': 'position'}),
        'interdms_ABAB_position_category': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'position', 'feature_2': 'category'}),
        'interdms_ABAB_position_identity': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'position', 'feature_2': 'identity'}),
        'interdms_ABAB_position_position': (
            InterDMSDataset, {'pattern': 'ABAB', 'feature_1': 'position', 'feature_2': 'position'}),
        'interdms_ABBA_category_category': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'category', 'feature_2': 'category'}),
        'interdms_ABBA_category_identity': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'category', 'feature_2': 'identity'}),
        'interdms_ABBA_category_position': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'category', 'feature_2': 'position'}),
        'interdms_ABBA_identity_category': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'identity', 'feature_2': 'category'}),
        'interdms_ABBA_identity_identity': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'identity', 'feature_2': 'identity'}),
        'interdms_ABBA_identity_position': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'identity', 'feature_2': 'position'}),
        'interdms_ABBA_position_category': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'position', 'feature_2': 'category'}),
        'interdms_ABBA_position_identity': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'position', 'feature_2': 'identity'}),
        'interdms_ABBA_position_position': (
            InterDMSDataset, {'pattern': 'ABBA', 'feature_1': 'position', 'feature_2': 'position'}),
    },
    'ctxdms': {
        'ctxdms_category_identity_position': (
            CtxDMSDataset,
            {"features": ["category", "identity", "position"]}
        ),
        'ctxdms_position_category_identity': (
            CtxDMSDataset,
            {"features": ["position", "category", "identity"]}
        ),
        'ctxdms_position_identity_category': (
            CtxDMSDataset,
            {"features": ["position", "identity", "category"]}
        ),
        'ctxdms_identity_position_category': (
            CtxDMSDataset,
            {"features": ["identity", "position", "category"]}
        )
    }
}

ALL_TASKS = {
    task_name: task_def
    for task_group in TASK_DATALOADERS.values()
    for task_name, task_def in task_group.items()
}

TASK_IDX_TASK_TYPE: dict[int, HvMTaskDataset] = dict()
for i in range(1, 44):
    if i < 10:
        TASK_IDX_TASK_TYPE[i] = NBackDataset
    elif i < 37:
        TASK_IDX_TASK_TYPE[i] = InterDMSDataset
    elif i < 41:
        TASK_IDX_TASK_TYPE[i] = CtxDMSDataset
    else:
        TASK_IDX_TASK_TYPE[i] = DMSDataset


class MultiTaskDataset(HvMTaskDataset, IterableDataset):
    """Iterable dataset that interleaves samples from all 43 tasks.

    Total ``dataset_size`` is split evenly across task groups and their subtasks.
    The trials of every subtask are regenerated (``reset``) at the start of each
    epoch, for the validation set as well as the training set, and sharded across
    DataLoader workers.
    """

    def __init__(
            self,
            hvm_loader,
            dataset_size=128,
            task_weights=None,
    ):
        super().__init__(
            hvm_loader=hvm_loader,
            dataset_size=dataset_size,
            task_len=6,
        )
        # multitask datasets use self.datasets to create trials
        self.dataset = None

        dataloaders_dict = ALL_TASKS
        self.datasets = dict()
        self.task_names = list(dataloaders_dict.keys())
        self.dataloaders_dict = dataloaders_dict

        self.task_dataset_size = self.get_dataset_size_per_task(dataset_size)
        for task_name, (DatasetClass, kwargs) in dataloaders_dict.items():
            dataset_kwargs = kwargs.copy()
            dataset_kwargs.update({
                'hvm_loader': hvm_loader,
                'pad_to': 6,
                'dataset_size': self.task_dataset_size[task_name],
            })
            # Instantiate the dataset
            self.datasets[task_name] = DatasetClass(**dataset_kwargs)

        # Calculate the maximum length among all datasets
        sizes = {name: len(ds) for name, ds in self.datasets.items()}
        assert sum(sizes.values()) == dataset_size, "Subtask sizes must sum to total"
        # Calculate weights based on dataset sizes for balanced sampling
        if task_weights is None:
            task_weights = {name: size / dataset_size for name, size in sizes.items()}
        self.task_weights = task_weights

    def get_dataset_size_per_task(self, total_dataset_size):
        """
        Evenly distribute total_dataset_size across task groups (e.g., 'dms', 'nback'),
        then evenly among subtasks within each group.
        """

        # Group tasks by task group prefix (e.g., 'dms', 'nback', etc.)
        task_group_dict = {task_group: dict() for task_group in TASK_DATALOADERS.keys()}
        task_groups = set()
        for task_name, v in self.dataloaders_dict.items():
            task_group = task_name.split('_')[0]
            if 'back' in task_group:
                task_group = 'nback'
            task_group_dict[task_group][task_name] = v
            task_groups.add(task_group)
        task_group_dict = {k: v for k, v in task_group_dict.items() if k in task_groups}

        num_groups = len(task_groups)
        base_group_size = total_dataset_size // num_groups
        group_remainder = total_dataset_size % num_groups

        task_dataset_size = {task: 0 for task in self.dataloaders_dict.keys()}
        for i, (group, subtasks) in enumerate(sorted(task_group_dict.items())):
            group_size = base_group_size + (1 if i < group_remainder else 0)
            num_subtasks = len(subtasks)
            base_subtask_size = group_size // num_subtasks
            subtask_remainder = group_size % num_subtasks

            for j, task_name in enumerate(subtasks):
                task_size = base_subtask_size + (1 if j < subtask_remainder else 0)
                task_dataset_size[task_name] = task_size
        return task_dataset_size

    def build_pairs(self):
        """Return a flat list of ``(task_name, sample_index)`` keys for lazy per-item iteration."""
        flat = list()
        for task_name, ds in self.datasets.items():
            n = len(ds)
            idxs = np.arange(n)
            flat.extend((task_name, int(i)) for i in idxs)
        return flat

    def __len__(self):
        return self.dataset_size

    def __iter__(self):
        wi = get_worker_info()

        for ds in self.datasets.values():
            if hasattr(ds, 'reset'):
                ds.reset()
        flat = self.build_pairs()

        N = len(flat)
        g = torch.Generator(device='cpu')
        g.manual_seed(torch.initial_seed() % (2 ** 63 - 1))
        perm = torch.randperm(N, generator=g).tolist()

        if wi is not None:
            w, i = wi.num_workers, wi.id
            base, r = N // w, N % w
            start = i * base + min(i, r)
            end = start + base + (1 if i < r else 0)
            perm = perm[start:end]

        for k in perm:
            task_name, idx = flat[k]
            yield self.datasets[task_name][idx]

    @staticmethod
    def get_stimuli_pair(trial_info, sort_pair=True):
        """Dispatch to the concrete task class's ``get_stimuli_pair`` based on ``task_id``."""
        task_id = trial_info['task_id']
        return TASK_IDX_TASK_TYPE[task_id].get_stimuli_pair(trial_info, sort_pair)
