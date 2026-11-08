"""Verifies seeded evaluation noise for :class:`DimMatchCNNRNN`.

``noise_seeds`` (one seed per trial, via :func:`models.model.gaussian_noise_like`) must
make a trial's input noise reproducible and independent of batch order and
composition, while the default ``noise_seeds=None`` keeps drawing fresh noise from
the global RNG as in training.
"""

import unittest

import torch

from models.model import CNNRNN
from models.dim_match import DimMatchCNNRNN, build_projection
from utils.const import INPUT_SIZE, OUTPUT_SIZE, TASK_NAME_TASK_INDEX

LABEL_SIZE = len(TASK_NAME_TASK_INDEX)
EMBEDDING_SIZE = INPUT_SIZE - LABEL_SIZE  # 85
N_CONT = 4


def make_batch(n, seed=0):
    """``n`` 6-frame trials: stimulus frames 0, 2, 4 (3-hot plus pose), blank frames 1, 3, 5."""
    g = torch.Generator().manual_seed(seed)
    emb = torch.zeros(n, 6, EMBEDDING_SIZE)
    idx = torch.arange(n)
    for t in (0, 2, 4):
        emb[idx, t, torch.randint(0, 8, (n,), generator=g)] = 1
        emb[idx, t, 8 + torch.randint(0, 64, (n,), generator=g)] = 1
        emb[idx, t, 72 + torch.randint(0, 9, (n,), generator=g)] = 1
        emb[:, t, -N_CONT:] = 0.4 * torch.randn(n, N_CONT, generator=g)
    task = torch.zeros(n, LABEL_SIZE)
    task[:, 2] = 1
    return emb, task


class TestDimMatchSeededNoise(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        core = CNNRNN(hidden_size=32, output_size=OUTPUT_SIZE, label_size=LABEL_SIZE,
                      seq_model_type='LSTM', num_heads=8, no_backbone=True, feat_dim=16, num_input_tokens=2)
        self.model = DimMatchCNNRNN(core=core, projection=build_projection(EMBEDDING_SIZE, 2, 16, 2025),
                                    n_cont=N_CONT, noise_scale=0.35, noise_pose_k=0.5).eval()
        self.emb, self.task = make_batch(16)
        self.seeds = list(range(100, 116))

    def test_reproducible(self):
        a = self.model(self.emb, self.task, noise_seeds=self.seeds)[0]
        self.assertTrue(torch.equal(a, self.model(self.emb, self.task, noise_seeds=self.seeds)[0]))
        other = self.model(self.emb, self.task, noise_seeds=[s + 1000 for s in self.seeds])[0]
        self.assertFalse(torch.allclose(a, other))

    def test_batch_independent(self):
        a = self.model(self.emb, self.task, noise_seeds=self.seeds)[0]
        perm = torch.randperm(16)
        c = self.model(self.emb[perm], self.task[perm], noise_seeds=[self.seeds[i] for i in perm])[0]
        self.assertTrue(torch.allclose(a[perm], c, atol=1e-6))
        d = self.model(self.emb[:5], self.task[:5], noise_seeds=self.seeds[:5])[0]
        self.assertTrue(torch.allclose(a[:5], d, atol=1e-6))

    def test_unseeded_draws_fresh_noise(self):
        torch.manual_seed(1)
        a = self.model(self.emb, self.task)[0]
        self.assertFalse(torch.allclose(a, self.model(self.emb, self.task)[0]))
        torch.manual_seed(1)
        self.assertTrue(torch.equal(a, self.model(self.emb, self.task)[0]))


if __name__ == '__main__':
    unittest.main()
