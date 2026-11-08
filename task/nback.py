"""N-back task variant built on the shared HvM dataset base.

Each frame after the first ``n`` is compared against the frame ``n`` steps back
on a chosen feature (category / identity / position), yielding a per-frame
match/non-match action. Task ids 1-9 encode ``n in {1,2,3}`` crossed with the
three features.
"""

from typing import Literal

from task.base import HvMTaskDataset
from utils.helper import get_one_hot


class NBackDataset(HvMTaskDataset):
    """N-back matching over HvM stimuli for a single ``(n, feature)`` combination."""

    def __init__(
            self,
            hvm_loader,
            dataset_size: int = 128,
            feature: Literal["category", "identity", "position"] = "category",
            nback_n: int = 1,
            pad_to: int = 0,
            task_index_base_value: int = 0,
            total_tasks: int = 43,
    ):
        task_len = max(6, pad_to)
        super(NBackDataset, self).__init__(
            dataset_size=dataset_size,
            hvm_loader=hvm_loader,
            task_len=task_len,
        )

        self.feature = feature
        self.nback_n = nback_n
        self.task_index = get_one_hot(
            {
                "1-position": task_index_base_value + 1,
                "1-identity": task_index_base_value + 2,
                "1-category": task_index_base_value + 3,
                "2-position": task_index_base_value + 4,
                "2-identity": task_index_base_value + 5,
                "2-category": task_index_base_value + 6,
                "3-position": task_index_base_value + 7,
                "3-identity": task_index_base_value + 8,
                "3-category": task_index_base_value + 9,
            }[f"{self.nback_n}-{self.feature}"],
            total=total_tasks
        )

    def _reset(self, i):
        self.dataset[i].zero_()
        self.actions[i] = 2

        buffer = []

        for j in range(self.nback_n):
            # randomly set the first nback_n frames
            obj = self._set_random(self.dataset[i, j])
            buffer += [obj]
        for j in range(self.nback_n, self.task_len):
            # set the rest of the frames according to the nback
            ref = buffer[j - self.nback_n]
            self.actions[i, j], *obj = self._set_data(self.dataset[i, j], *ref)
            buffer.append(obj)

    def reset(self):
        super().reset()
        for i in range(self.dataset_size):
            self._reset(i)

    @classmethod
    def get_stimuli_pair(cls, trial_info, sort_pair=True):
        """Return one ``(frame-n, frame)`` HvM pair per action frame, plus the compared feature."""
        row_idx_list = trial_info['row_idx']
        label = trial_info['trial_label']
        task_id = trial_info['task_id']

        assert task_id in range(1, 10)
        base = 0
        rel = task_id - base

        idx0 = rel - 1  # 0..8
        nback_n = (idx0 // 3) + 1  # 1,2,3
        feat_idx = idx0 % 3  # 0->position,1->identity,2->category
        feat_map = {0: "position", 1: "identity", 2: "category"}
        feature = feat_map[feat_idx]

        n = len(label)
        assert n == len(row_idx_list), f"Length mismatch between labels, {label} and row indices {row_idx_list}"

        action_positions = [i for i in range(n) if label[i] != 2]
        sample_positions = [i for i in range(n) if row_idx_list[i] != -1]

        pairs = []
        features = []
        for pos in action_positions:
            sample_pos = pos - nback_n
            if sort_pair:
                pair = tuple(sorted([row_idx_list[sample_pos], row_idx_list[pos]]))
            else:
                pair = (row_idx_list[sample_pos], row_idx_list[pos])
            pairs.append(pair)
            features.append(feature)
        return pairs, features
