"""Base match/non-match task dataset definitions.

Defines :class:`TaskDataset`, the abstract generator of match/non-match
embedding sequences (one discrete feature, either category, identity or
position, repeats to signal a match or is resampled), and :class:`HvMTaskDataset`,
which draws the stimuli of those sequences from the HvM image metadata. Concrete
task variants (DMS, inter-DMS, context-DMS, n-back) subclass the latter.
"""

import random
from typing import Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.stim_io import HvMImageLoader, decode_embedding


class TaskDataset(Dataset):
    """
    Abstract dataset class for generating embedding sequences for matching tasks.
    Each task is a sequence of trials where one discrete feature (e.g. category, identity, etc.)
    may repeat, indicating a match, and others vary. Subclasses supply the stimulus
    sampling (``_set_random`` and the ``available_*`` methods).

    Attributes:
        dataset_size: Number of tasks in the dataset.
        task_len: Number of items per task.
        category_size, identity_size, position_size: Discrete feature vocab sizes.
        view_size, size_dim: Continuous feature dimensions.
        feature: The feature used to define "match" conditions.
    """

    def __init__(
            self,
            dataset_size: int,
            task_len: int = 6,
            category_size: int = 8,
            identity_size: int = 8,
            position_size: int = 9,
            view_size: int = 3,
            size_dim: int = 1,
            feature: str = 'category',
    ):
        assert position_size in [4, 9], 'position size should be 2x2 or 3x3 grid'
        self.feature = feature
        self.task_index = None
        self.task_len = task_len
        self.dataset_size = dataset_size

        # Discrete feature vocab sizes
        self.category_size = category_size
        self.identity_size = identity_size
        self.position_size = position_size
        # Continuous feature dimensions
        self.view_size = view_size
        self.size_dim = size_dim

        # Total embedding size (discrete + continuous parts)
        self.embedding_size = (
                category_size +
                category_size * identity_size +
                position_size +
                view_size +
                size_dim
        )

        self.cont_size = (dataset_size, task_len, view_size + size_dim)
        self.one_hot_size = (dataset_size, task_len, self.embedding_size - (view_size + size_dim))

        one_hot = torch.zeros(self.one_hot_size)
        self.dataset = torch.cat([one_hot, torch.zeros(self.cont_size)], dim=-1)

        # Default action is "no action" (2)
        self.actions = torch.full((dataset_size, task_len), 2, dtype=torch.long)  # 2 = no action

    @staticmethod
    def _sample_from(choices, exclude: Optional[int] = None) -> int:
        """Return random element from `choices` excluding `exclude` (if present)."""
        if exclude is None:
            return random.choice(choices)
        filtered = set(choices)
        filtered.discard(exclude)
        if not filtered:
            raise RuntimeError(f"No alternative choice available (excluded={exclude})")
        return random.choice(list(filtered))

    def _get_new(
            self,
            prev: int,
            category: Optional[int] = None,
            identity: Optional[int] = None
    ) -> int:
        """
        Return a new available value (1-based) for the current feature, excluding `prev`.
        `category` is used when sampling identities (to pick identities available in that category).
        """
        if self.feature == "category":
            choices = self.available_categories()
            return int(self._sample_from(choices, exclude=prev))

        elif self.feature == "identity":
            # sample among identities available for the given category.
            choices = self.available_identities(category)
            return int(self._sample_from(choices, exclude=prev))
        else:  # position
            choices = self.available_positions(category, identity)
            return int(self._sample_from(choices, exclude=prev))

    def _set_data(
            self,
            data: torch.Tensor,
            category: int, identity: int, position: int,
            view=None, size=None,
    ):
        """
       Write one trial into the task sequence.
       With 50% chance, repeat a feature value (action = 1), else sample a new one (action = 0).
       """
        action = 0
        if random.random() < 0.5:  # match
            action = 1
            if self.feature == 'category':
                category, identity, position, view, size = self._set_random(
                    data, category=category
                )
            elif self.feature == 'identity':
                category, identity, position, view, size = self._set_random(
                    data, category=category, identity=identity
                )
            elif self.feature == 'position':
                category, identity, position, view, size = self._set_random(
                    data, position=position
                )
        else:  # no match, sample new feature
            if self.feature == 'category':
                new_category = self._get_new(category, self.category_size)
                category, identity, position, view, size = self._set_random(
                    data, category=new_category
                )
            elif self.feature == 'identity':
                new_category = random.choice(self.available_categories())
                if new_category == category:
                    new_identity = self._get_new(
                        identity, category=new_category
                    )
                else:
                    new_identity = self._sample_from(self.available_identities(category=new_category))
                category, identity, position, view, size = self._set_random(
                    data, category=new_category, identity=new_identity
                )
            elif self.feature == 'position':
                rand_category = random.choice(self.available_categories())
                rand_identity = self._sample_from(
                    self.available_identities(category=rand_category)
                )
                new_position = self._get_new(
                    position, category=rand_category, identity=rand_identity,
                )
                category, identity, position, view, size = self._set_random(
                    data,
                    category=rand_category,
                    identity=rand_identity,
                    position=new_position
                )
        return action, category, identity, position, view, size

    def _encode_discrete(self, data: torch.Tensor, category: int, identity: int, position: int):
        """
        Encode category, identity, and position into one-hot vectors in `data`.
        """
        # category one-hot
        data[:self.category_size].zero_()
        data[category - 1] = 1

        # identity block
        identity_start = self.category_size
        identity_end = identity_start + self.category_size * self.identity_size
        data[identity_start:identity_end].zero_()
        identity_offset = identity_start + (category - 1) * self.identity_size + identity - 1
        data[identity_offset] = 1

        position_start = identity_end
        position_end = position_start + self.position_size
        data[position_start:position_end].zero_()
        data[position_start + position - 1] = 1
        return

    def _encode_continuous(self, data: torch.Tensor, view: Tuple[float, ...], size: float):
        """
        Encode view and size features into the final segment of the embedding vector.
        """
        start = self.embedding_size - (self.view_size + self.size_dim)
        if self.view_size > 0:
            data[start:start + self.view_size] = torch.tensor(view, dtype=data.dtype)
        if self.size_dim == 1:
            data[start + self.view_size] = size

    def decode_embedding(self, embedding: torch.Tensor):
        """Decode a single embedding vector back to (category, identity, position, view, size)."""
        return decode_embedding(
            embedding,
            self.category_size,
            self.identity_size,
            self.position_size,
            self.view_size,
            self.size_dim
        )

    @classmethod
    def get_stimuli_pair(cls, trial_info, sort_pair=True):
        """
        Given trial info dict, return stimulus pairs
        trial_info: {
        'row_idx': List[int], the row indices in the HvM dataset
        'trial_label': List[int], 0/1/2 for no-match/match/miss
        'task_id': int, the task index,
        'position': List[int], the positions of the stimuli in the grid
        'category': List[int], the categories of the stimuli
        'identity': List[int], the identities of the stimuli
        }
        """
        raise NotImplementedError

    @classmethod
    def get_ordered_action(cls, action, task_id=None):
        return action

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        data = self.dataset[idx]
        action = self.actions[idx]
        return data, action, torch.tensor(self.task_index)

    def reset(self):
        """Clear all sequences and actions back to zeros / the default no-match action."""
        self.dataset = torch.zeros((self.dataset_size, self.task_len, self.embedding_size))
        self.actions = torch.full((self.dataset_size, self.task_len), 2, dtype=torch.long)


class HvMTaskDataset(TaskDataset):
    """
    TaskDataset that draws from real HvM images/metadata.

    - `_set_random`: samples an HvM row that matches any specified discrete features.
    - `_get_new`: selects a new discrete feature value that is valid in the HvM cache.
    - `_set_data`: inherited, builds match/non-match sequences using `_set_random` and `_get_new`.
    """

    def __init__(
            self,
            hvm_loader: HvMImageLoader,
            dataset_size: int,
            task_len: int = 6,
            feature: str = "category",
            grid_size: int = 3,
    ):
        # Prepare the loader cache (positions, continuous features, optional images)
        if not hasattr(hvm_loader, '_task_cache'):
            hvm_loader.prepare_for_tasks(grid_size=grid_size)
        self.cache = hvm_loader._task_cache
        self.hvm_loader = hvm_loader
        self.feature = feature

        super().__init__(
            dataset_size=dataset_size,
            task_len=task_len,
            category_size=self.cache.category_size,
            identity_size=self.cache.identity_size,
            position_size=self.cache.position_size,
            view_size=self.cache.view_size,
            size_dim=self.cache.size_dim,
            feature=feature,
        )

        # Override embedding size
        self.embedding_size = self.cache.embedding_size
        # Precompute arrays for efficient masking
        self._cats = self.cache.cat_1b
        self._ids = self.cache.id_1b
        self._poses = self.cache.pos_1b
        self._views = self.cache.views
        self._sizes = self.cache.sizes
        self._vars = self.cache.vars
        self._valid_triples = self.cache.valid_triples
        self._row_indices = np.arange(len(self._cats))

        self.categories = [
            c for c in range(1, self.category_size + 1)
        ]
        self.ids = {c: list() for c in range(1, self.category_size + 1)}
        self.pos = self.cache.catid_to_positions

        self.global_pos = set(range(1, self.position_size + 1))
        for c in self.categories:
            for i in range(1, self.identity_size + 1):
                if np.where((self._ids == i) & (self._cats == c))[0].size > 0:
                    self.ids[c].append(i)
                key = (c, i)
                if key in self.pos:
                    self.global_pos &= set(self.pos[key])

    def available_categories(self):
        return self.categories

    def available_identities(self, category: Optional[int] = None):
        if category is None:
            return np.unique(self._ids)
        return self.ids[category]

    def available_positions(self, category=None, identity=None):
        if category is None or identity is None:
            print('sampling random pos')
            return self.global_pos
        return self.pos[(category, identity)]

    def _set_random(
            self,
            data: torch.Tensor,
            category: Optional[int] = None,
            identity: Optional[int] = None,
            position: Optional[int] = None,
            view=None,
            size=None
    ) -> Tuple[int, int, int, tuple, float]:
        """
        Samples a single HvM row matching any fixed discrete features.
        Leaves unspecified ones to vary.
        """
        # Start with all rows
        mask = np.ones_like(self._row_indices, dtype=bool)

        if category is not None:
            mask &= (self._cats == category)
        if identity is not None:
            mask &= (self._ids == identity)
        if position is not None:
            mask &= (self._poses == position)

        candidates = self._row_indices[mask]
        if len(candidates) == 0:
            raise RuntimeError(
                f"No HvM images match category={category}, identity={identity}, position={position}"
            )

        cand_vars = self._vars[candidates]
        weights = np.ones_like(cand_vars, dtype=np.float32)
        weights[cand_vars == 6] = 0.5  # pick var6 2x less often

        probs = weights / weights.sum()
        idx = int(np.random.choice(candidates, p=probs))

        # Extract features from that row
        category, identity, position = int(self._cats[idx]), int(self._ids[idx]), int(self._poses[idx])
        view = tuple(self._views[idx]) if self._views is not None else ()
        size = float(self._sizes[idx]) if self._sizes is not None else 0.0
        assert (category, identity, position) in self._valid_triples
        # Encode into data
        self._encode_discrete(data, category, identity, position)
        self._encode_continuous(data, view, size)

        return category, identity, position, view, size
