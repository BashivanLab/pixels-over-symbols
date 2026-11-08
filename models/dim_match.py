"""Dimensionality-matched abstract model: one-hot stimulus embeddings lifted to vision-token shape."""
import torch
import torch.nn as nn

from models.model import CNNRNN, gaussian_noise_like

# Scale of the view/size features (min-max normalized to [-1, 1]); dividing by it
# gives the pose z-scores whose RMS sets the per-stimulus noise level.
VIEW_SIGMA = 0.4

# Mean pose z-score RMS over the stimulus distribution, measured over 30k sampled
# stimuli. The pose term is centred here so that noise_pose_k redistributes noise
# across poses without shifting the average noise level: a typical stimulus has
# pose_rms ~= 0.72, so centring on 1.0 would lower the mean sigma as k grows.
POSE_RMS_MEAN = 0.715


def build_projection(embedding_size, num_tokens, upsample_dim, seed):
    """Fixed (non-trainable) random lift ``embedding -> (num_tokens, upsample_dim)``.

    Entries are ``N(0, 1)`` scaled by ``1/sqrt(embedding_size)``, so a token channel
    of a stimulus with embedding ``e`` has std ``||e|| / sqrt(embedding_size)`` --
    ~0.2 here, since ``e`` is 3-hot plus 4 small pose channels rather than a dense
    unit-variance vector. That 0.2 is what ``noise_scale`` competes against, so
    0.2 is roughly unit SNR.
    The same projection must be shared across train/val (and reproducible), so it
    is seeded with ``2025 + rep_index`` by the caller and stored as a model buffer.
    """
    g = torch.Generator().manual_seed(seed)
    proj = torch.randn(embedding_size, num_tokens, upsample_dim, generator=g)
    proj = proj / (embedding_size ** 0.5)
    return proj


class DimMatchCNNRNN(nn.Module):
    """CNNRNN fed upsampled + noised abstract embeddings instead of vision tokens.

    ``forward`` lifts each ``(B, T, E)`` stimulus embedding to ``(B, T, L, D)``
    tokens via the fixed ``projection`` buffer, adds pose-scaled Gaussian noise
    (fresh every call, so the two presentations of a "match" differ, and present
    at eval time too since the task is inherently noisy), and hands the result to
    a no-backbone :class:`CNNRNN`. Blank (all-zero) frames project to zero tokens
    and receive no noise.

    ``noise_scale`` (the noise level s in the paper) is an absolute std: clean
    token channels have std ~0.2, so 0.2 is roughly unit SNR. With
    ``noise_pose_k > 0`` the std also grows with the pose extremity (view/size)
    of the stimulus, so it models viewing difficulty and not position or
    pair-wise confusability.
    """

    def __init__(
            self,
            core: CNNRNN,
            projection: torch.Tensor,
            n_cont: int,
            noise_scale: float,
            noise_pose_k: float,
            view_sigma: float = VIEW_SIGMA
    ):
        super().__init__()
        self.core = core
        self.register_buffer('projection', projection)  # (E, L, D), fixed
        self.n_cont = n_cont
        self.noise_scale = noise_scale
        self.noise_pose_k = noise_pose_k
        self.view_sigma = view_sigma

    def forward(self, emb, task_index, *args, noise_seeds=None, **kwargs):
        # emb: (B, T, E) abstract embedding; blank frames are all-zero.
        # noise_seeds: None for fresh noise, or one seed per trial (seeded evaluation).
        tokens = torch.einsum('bte,eld->btld', emb, self.projection)  # (B, T, L, D)

        # gate blank frames off
        frame_scale = (emb.abs().sum(dim=-1) > 0).to(tokens.dtype)  # (B, T)

        # pose extremity ~ RMS z-score over the continuous (view/size) channels.
        # Extreme rotations/scales get more noise, so sigma tracks how hard the
        # stimulus is to view. Centred on POSE_RMS_MEAN so k does not also move the
        # average noise level; k = 0 leaves sigma constant at noise_scale.
        cont = emb[..., -self.n_cont:] / self.view_sigma
        pose_rms = cont.pow(2).mean(dim=-1).sqrt()  # (B, T)
        sigma = self.noise_scale * (1.0 + self.noise_pose_k * (pose_rms - POSE_RMS_MEAN))
        sigma = (sigma.clamp(min=0.0) * frame_scale)[..., None, None]  # (B, T, 1, 1)

        tokens = tokens + gaussian_noise_like(tokens, noise_seeds) * sigma
        return self.core(tokens, task_index, *args, **kwargs)
