"""Concrete delayed-match-to-sample (DMS) task variants.

Each dataset builds fixed-length trial sequences and marks which frames carry a
response (action 0 = non-match, 1 = match, 2 = no action):

- :class:`DMSDataset`: a single sample frame and a single test frame compared
  on one feature (category / identity / position).
- :class:`InterDMSDataset`: two interleaved DMS problems whose sample/test
  frames follow an AABB, ABAB, or ABBA temporal pattern.
- :class:`CtxDMSDataset`: a context/branching DMS where the outcome of the
  first comparison selects which feature the second comparison uses.

Each task carries a one-hot ``task_index`` encoding its id within the 43-task
space, so a single model can be trained across all variants.
"""

import random
from typing import Literal, List

from task.base import HvMTaskDataset
from utils.helper import get_one_hot


class DMSDataset(HvMTaskDataset):
    """Single-comparison DMS: sample frame vs. test frame on one feature.

    Task ids 41 (position), 42 (identity), 43 (category). The first frame holds
    the sample; the last frame is the test and receives the match/non-match
    action, with the middle frames padded as no-action.
    """

    def __init__(
            self,
            hvm_loader,
            dataset_size: int = 128,
            feature: Literal["category", "identity", "position"] = "category",
            pad_to: int = 0,
            task_index_base_value: int = 40,
            total_tasks: int = 43,
    ):
        task_len = max(2, pad_to)
        super().__init__(
            hvm_loader=hvm_loader,
            dataset_size=dataset_size,
            task_len=task_len,
            feature=feature,
        )
        self.feature = feature
        self.task_index = get_one_hot(
            {
                "position": task_index_base_value + 1,
                "identity": task_index_base_value + 2,
                "category": task_index_base_value + 3,
            }[feature], total=total_tasks
        )

    def _reset(self, i):
        # initialize the frames
        self.dataset[i].zero_()
        self.actions[i] = 2  # No action

        # init the first frame
        obj = self._set_random(self.dataset[i, 0])

        # Set final frame
        self.actions[i, self.task_len - 1], *_ = self._set_data(
            self.dataset[i, self.task_len - 1], *obj
        )
        return

    def reset(self):
        super().reset()
        for i in range(self.dataset_size):
            self._reset(i)
        return

    @classmethod
    def get_stimuli_pair(cls, trial_info, sort_pair=True):
        """Return the ``[(sample_row, test_row)]`` HvM pair and compared feature for a DMS trial."""
        row_idx_list = trial_info['row_idx']
        label = trial_info['trial_label']
        task_id = trial_info['task_id']
        assert task_id in [41, 42, 43]
        feature = 'position' if task_id == 41 else 'identity' if task_id == 42 else 'category'

        n = len(label)
        assert n == len(row_idx_list)
        action_positions = [i for i in range(n) if label[i] != 2]
        stim_positions = [i for i in range(n) if row_idx_list[i] != -1]

        assert len(action_positions) == 1, 'DMS should only have 1 action frame'
        assert len(stim_positions) == 2, 'DMS should only have 2 stimuli frames'
        if sort_pair:
            stim_pairs = sorted([row_idx_list[stim_positions[0]], row_idx_list[stim_positions[1]]])
            stim_pairs = tuple(stim_pairs)
        else:
            stim_pairs = (row_idx_list[stim_positions[0]], row_idx_list[stim_positions[1]])
        return [stim_pairs], [feature]


class InterDMSDataset(HvMTaskDataset):
    """Two interleaved DMS problems (A and B) with a temporal pattern.

    ``pattern`` places the two sample and two test frames as AABB, ABAB, or
    ABBA; problem A is compared on ``feature_1`` and problem B on ``feature_2``.
    Task ids span 10-36, encoded as ``10 + pattern*9 + f1*3 + f2``.
    """

    def __init__(
            self,
            hvm_loader,
            dataset_size: int = 128,
            feature_1: Literal["category", "identity", "position"] = "category",
            feature_2: Literal["category", "identity", "position"] = "category",
            pattern: Literal["AABB", "ABBA", "ABAB"] = "AABB",
            pad_to: int = 0,
            task_index_base_value: int = 10,
            total_tasks: int = 43,
    ):
        task_len = max(4, pad_to)
        super().__init__(
            hvm_loader=hvm_loader,
            dataset_size=dataset_size,
            task_len=task_len,
        )
        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.pattern = pattern
        self.task_index = get_one_hot(
            task_index_base_value +
            {"AABB": 0, "ABAB": 1, "ABBA": 2}[pattern] * 9 +
            {"category": 2, "identity": 1, "position": 0}[feature_1] * 3 +
            {"category": 2, "identity": 1, "position": 0}[feature_2],
            total=total_tasks
        )

    def _reset_two_tasks(self, i, a_stim, a_response, b_stim, b_response):
        self.feature = self.feature_1
        A_obj = self._set_random(self.dataset[i, a_stim])
        self.actions[i, a_response], *_ = self._set_data(
            self.dataset[i, a_response], *A_obj
        )

        self.feature = self.feature_2
        B_obj = self._set_random(self.dataset[i, b_stim])
        self.actions[i, b_response], *_ = self._set_data(
            self.dataset[i, b_response], *B_obj
        )

    def _reset_AABB(self, i):
        # initialize the frames
        self.dataset[i].zero_()
        self.actions[i] = 2  # No action

        a_stim = 0
        a_response, b_stim = sorted(random.sample(range(1, self.task_len - 1), 2))
        b_response = self.task_len - 1

        self._reset_two_tasks(i, a_stim, a_response, b_stim, b_response)

    def _reset_ABBA(self, i):
        # initialize the frames
        self.dataset[i].zero_()
        self.actions[i] = 2  # No action

        a_stim = 0
        b_stim, b_response = sorted(random.sample(range(1, self.task_len - 1), 2))
        a_response = self.task_len - 1

        self._reset_two_tasks(i, a_stim, a_response, b_stim, b_response)

    def _reset_ABAB(self, i):
        # initialize the frames
        self.dataset[i].zero_()
        self.actions[i] = 2  # No action

        a_stim = 0
        b_stim, a_response = sorted(random.sample(range(1, self.task_len - 1), 2))
        b_response = self.task_len - 1

        self._reset_two_tasks(i, a_stim, a_response, b_stim, b_response)

    def reset(self):
        super().reset()
        reset_func = {
            "AABB": self._reset_AABB,
            "ABBA": self._reset_ABBA,
            "ABAB": self._reset_ABAB,
        }[self.pattern]
        for i in range(self.dataset_size):
            reset_func(i)

    @classmethod
    def get_interdms_type(cls, task_id):
        """Decode an inter-DMS task id (10-36) into ``(pattern, feature_1, feature_2)``."""
        assert task_id in range(10, 37)
        base = 10
        rel = task_id - base
        pattern_idx = rel // 9
        rem = rel % 9
        f1_idx = rem // 3
        f2_idx = rem % 3

        pattern_map = {0: "AABB", 1: "ABAB", 2: "ABBA"}
        feat_map = {0: "position", 1: "identity", 2: "category"}

        pattern = pattern_map[pattern_idx]
        feature_1 = feat_map[f1_idx]
        feature_2 = feat_map[f2_idx]
        return pattern, feature_1, feature_2

    @classmethod
    def get_ordered_action(cls, action, task_id):
        """Reorder the two actions into (A, B) problem order; ABBA swaps them back."""
        assert len(action) == 2, 'InterDMS should only have 2 actions'
        pattern, feature_1, feature_2 = cls.get_interdms_type(task_id)
        if pattern == 'ABBA':
            return [action[1], action[0]]
        return action

    @classmethod
    def get_stimuli_pair(cls, trial_info, sort_pair=True):
        """Return the two ``(sample, test)`` HvM pairs and their features, de-interleaved by pattern."""
        row_idx_list = trial_info['row_idx']
        label = trial_info['trial_label']
        task_id = trial_info['task_id']
        pattern, feature_1, feature_2 = cls.get_interdms_type(task_id)
        n = len(label)
        assert n == len(row_idx_list)

        action_positions = [i for i in range(n) if label[i] != 2]
        sample_positions = [i for i in range(n) if row_idx_list[i] != -1]
        assert len(sample_positions) == 4, 'interdms should have 4 stimuli'
        assert sample_positions[0] == 0, 'first frame of interdms always has stimulus'
        assert sample_positions[-1] == n - 1, 'last frame of interdms always has stimulus'
        assert len(action_positions) == 2

        pair_a, pair_b = list(), list()
        sp = sample_positions
        if pattern == 'AABB':
            pair_a = [row_idx_list[sp[0]], row_idx_list[sp[1]]]
            pair_b = [row_idx_list[sp[2]], row_idx_list[sp[3]]]
        elif pattern == 'ABAB':
            pair_a = [row_idx_list[sp[0]], row_idx_list[sp[2]]]
            pair_b = [row_idx_list[sp[1]], row_idx_list[sp[3]]]
        elif pattern == 'ABBA':  # ABBA
            pair_a = [row_idx_list[sp[0]], row_idx_list[sp[3]]]
            pair_b = [row_idx_list[sp[1]], row_idx_list[sp[2]]]
        else:
            raise RuntimeError()

        if sort_pair:
            pairs = [tuple(sorted(pair_a)), tuple(sorted(pair_b))]
        else:
            pairs = [tuple(pair_a), tuple(pair_b)]
        return pairs, [feature_1, feature_2]


class CtxDMSDataset(HvMTaskDataset):
    """Context-dependent (branching) DMS over three ordered features.

    The first comparison uses ``features[0]``; its match/non-match outcome
    selects whether the second comparison uses ``features[1]`` (on match) or
    ``features[2]`` (on non-match). Task ids span 37-40.
    """

    def __init__(
            self,
            hvm_loader,
            dataset_size: int = 128,
            features: List[Literal["category", "identity", "position"]] = ["category", "identity", "position"],
            pad_to: int = 0,
            task_index_base_value: int = 36,
            total_tasks: int = 43,
    ):
        task_len = max(3, pad_to)
        super().__init__(
            hvm_loader=hvm_loader,
            dataset_size=dataset_size,
            task_len=task_len,
        )
        self.features = features
        self.task_index = get_one_hot(
            task_index_base_value +
            {
                ("position", "category", "identity"): 1,
                ("position", "identity", "category"): 2,
                ("identity", "position", "category"): 3,
                ("category", "identity", "position"): 4,
            }[tuple(features)]
        )

    def _reset(self, i):
        # initialize the frames
        self.dataset[i].zero_()
        self.actions[i] = 2  # No action

        a_stim = 0
        a_response = random.choice(range(1, self.task_len - 1))
        b_response = self.task_len - 1

        ctx_obj = self._set_random(self.dataset[i, a_stim])
        self.feature = self.features[0]
        # the first comparison has no response; its outcome selects the second feature
        ctx_action, *dms_obj = self._set_data(
            self.dataset[i, a_response],
            *ctx_obj
        )

        if ctx_action == 1:  # if the first DMS task is true, compare second feature
            self.feature = self.features[1]
            self.actions[i, b_response], *_ = self._set_data(
                self.dataset[i, b_response],
                *dms_obj
            )
        elif ctx_action == 0:  # if the first DMS task is false, compare third feature
            self.feature = self.features[2]
            self.actions[i, b_response], *_ = self._set_data(
                self.dataset[i, b_response],
                *dms_obj
            )

    def reset(self):
        super().reset()
        for i in range(self.dataset_size):
            self._reset(i)

    # we count the ctx as a triplet
    @classmethod
    def get_stimuli_pair(cls, trial_info, sort_pair=True):
        """Return the context-DMS stimulus triplet and the two features actually compared on this trial."""
        row_idx_list = trial_info['row_idx']
        label = trial_info['trial_label']
        task_id = trial_info['task_id']
        position = trial_info['position']
        category = trial_info['category']
        identity = trial_info['identity']

        assert task_id in range(37, 41)
        base = 36
        rel = task_id - base

        rev_map = {
            1: ["position", "category", "identity"],
            2: ["position", "identity", "category"],
            3: ["identity", "position", "category"],
            4: ["category", "identity", "position"],
        }
        features = rev_map[rel]

        n = len(label)
        assert n == len(row_idx_list)

        action_positions = [i for i in range(n) if label[i] != 2]
        sample_positions = [i for i in range(n) if row_idx_list[i] != -1]
        assert len(sample_positions) == 3, 'ctxdms should have 3 stimuli'
        assert len(action_positions) == 1, 'ctxdms should have 1 action frame'

        sp = sample_positions
        pair = [row_idx_list[sp[0]], row_idx_list[sp[1]], row_idx_list[sp[2]]]
        if features[0] == 'position':
            if rel == 1:
                if position[sp[0]] == position[sp[1]]:
                    features.pop(-1)  # remove identity
                else:
                    features.pop(-2)  # remove category
            elif rel == 2:
                if position[sp[0]] == position[sp[1]]:
                    features.pop(-2)  # remove category
                else:
                    features.pop(-1)  # remove identity
        elif features[0] == 'identity':
            if identity[sp[0]] == identity[sp[1]]:
                features.pop(-1)
            else:
                features.pop(-2)
        else:  # category
            if category[sp[0]] == category[sp[1]]:
                features.pop(-1)
            else:
                features.pop(-2)

        if sort_pair:
            pairs = [tuple(sorted(pair))]
        else:
            pairs = [tuple(pair)]
        return pairs, features
