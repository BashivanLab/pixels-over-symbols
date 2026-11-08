"""Verifies --freeze_seq_model / --freeze_input_pathway / --freeze_fc_out on both substitution directions.

The models of the dimensionality-matched substitution experiment are:

- Natural: :class:`CNNRNN` over images (``train_natural_frozen.py``, Abs->Nat).
- Abstract: :class:`DimMatchCNNRNN` over dimensionality-matched abstract
  tokens (``train_abstract_dim_match.py``, Nat->Abs), which wraps a no-backbone
  :class:`CNNRNN` as ``self.core``.

Because both run the same ``CNNRNN`` pipeline, the freeze flags must expose the
same trainable parameters on each side, otherwise an asymmetry in what can adapt
gets attributed to the substituted dynamics. Each flag is checked four ways:

1. ``requires_grad`` is False on exactly the intended parameters,
2. those parameters are bit-identical after a real backward + optimizer step,
3. the frozen submodules report ``training is False`` after ``model.train()``, and
4. the two models agree on which parameters remain trainable.
"""

import copy
import io
import unittest

import torch
import torch.nn.functional as F

from models.model import CNNRNN
from models.dim_match import DimMatchCNNRNN, build_projection
from utils.const import INPUT_SIZE, OUTPUT_SIZE, TASK_NAME_TASK_INDEX

HIDDEN = 32
LABEL_SIZE = len(TASK_NAME_TASK_INDEX)
EMBEDDING_SIZE = INPUT_SIZE - LABEL_SIZE
# Match the real dim-match config: a ResNet50 emits 49 tokens of 2048-d, and
# DimMatchCNNRNN is sized to produce exactly that shape.
UPSAMPLE_DIM = 2048
NUM_TOKENS = 49
N_CONT = 4
CNN_TYPE = 'resnet50_clip.openai'
B, T = 2, 2

SEQ_PREFIXES = ('seq_model.',)
FC_OUT_PREFIXES = ('norm_out.', 'fc_out.')
# everything upstream of the core, which is what freeze_input_pathway covers
INPUT_PATHWAY_PREFIXES = ('proj.', 'attn_pool.', 'fc_in.', 'norm_in.')


def build_natural(**freeze_kwargs):
    """The Natural model: CNNRNN with a (always-frozen) vision backbone.

    The backbone weights do not matter here, so it is randomly initialized and
    the test needs no download.
    """
    return CNNRNN(
        hidden_size=HIDDEN, output_size=OUTPUT_SIZE, label_size=LABEL_SIZE,
        seq_model_type='LSTM', dropout=0.3, num_heads=8,
        cnn_type=CNN_TYPE, pretrained=False,
        **freeze_kwargs,
    )


def build_dim_match(noise_scale=0.5, **freeze_kwargs):
    """The dimensionality-matched Abstract model, built as train_abstract_dim_match() builds it."""
    core = CNNRNN(
        hidden_size=HIDDEN, output_size=OUTPUT_SIZE, label_size=LABEL_SIZE,
        seq_model_type='LSTM', dropout=0.3, num_heads=8,
        no_backbone=True, feat_dim=UPSAMPLE_DIM, num_input_tokens=NUM_TOKENS,
        **freeze_kwargs,
    )
    projection = build_projection(EMBEDDING_SIZE, NUM_TOKENS, UPSAMPLE_DIM, seed=2025)
    return DimMatchCNNRNN(
        core=core, projection=projection, n_cont=N_CONT,
        noise_scale=noise_scale, noise_pose_k=0.5,
    )


def natural_inputs():
    x = torch.randn(B, T, 3, 224, 224)
    task_index = torch.zeros(B, LABEL_SIZE)
    task_index[:, 0] = 1
    return x, task_index


def dim_match_inputs():
    x = torch.randn(B, T, EMBEDDING_SIZE)
    task_index = torch.zeros(B, LABEL_SIZE)
    task_index[:, 0] = 1
    return x, task_index


def core_of(model):
    """The CNNRNN inside either model."""
    return model.core if isinstance(model, DimMatchCNNRNN) else model


def tunable_named_parameters(model):
    """Parameters the freeze flags govern, i.e. excluding the vision backbone.

    The backbone is unconditionally frozen at construction and is not what these
    flags are about; including it would mask the comparison between the models.
    """
    for name, param in model.named_parameters():
        stripped = name[len('core.'):] if name.startswith('core.') else name
        if stripped.startswith('vision_backbone.'):
            continue
        yield stripped, param


def step_once(model, inputs):
    """One real forward/backward/AdamW step; returns params before the step.

    Returns ``None`` when nothing is trainable, which is a legitimate outcome for
    some flag combinations.
    """
    before = copy.deepcopy({k: v.detach().clone() for k, v in model.named_parameters()})
    trainable = [p for p in model.parameters() if p.requires_grad]
    if not trainable:
        return None
    opt = torch.optim.AdamW(trainable, lr=1e-2)
    model.train()
    x, task_index = inputs
    out, _, _ = model(x, task_index)
    labels = torch.randint(0, OUTPUT_SIZE, (B * T,))
    loss = F.cross_entropy(out.reshape(-1, OUTPUT_SIZE), labels)
    loss.backward()
    opt.step()
    return before


class FreezeFlagMixin:
    """Shared assertions; subclasses supply the builder and input factory."""

    build = None
    make_inputs = None

    def expected_frozen_prefixes(self, **flags):
        prefixes = ()
        if flags.get('freeze_seq_model'):
            prefixes += SEQ_PREFIXES
        if flags.get('freeze_fc_out'):
            prefixes += FC_OUT_PREFIXES
        if flags.get('freeze_input_pathway'):
            prefixes += INPUT_PATHWAY_PREFIXES
        return prefixes

    def check_flags(self, **flags):
        model = type(self).build(**flags)
        frozen_prefixes = self.expected_frozen_prefixes(**flags)

        # 1. requires_grad matches exactly the documented set.
        for name, param in tunable_named_parameters(model):
            should_freeze = bool(frozen_prefixes) and name.startswith(frozen_prefixes)
            self.assertEqual(
                not param.requires_grad, should_freeze,
                f"{flags}: {name} requires_grad={param.requires_grad}, expected frozen={should_freeze}",
            )

        # 2. Frozen params must not move under a real optimizer step.
        before = step_once(model, type(self).make_inputs())
        if before is None:
            return model
        for name, param in model.named_parameters():
            stripped = name[len('core.'):] if name.startswith('core.') else name
            if bool(frozen_prefixes) and stripped.startswith(frozen_prefixes):
                self.assertTrue(
                    torch.equal(before[name], param.detach()),
                    f"{flags}: frozen param {name} CHANGED after an optimizer step",
                )
        return model

    def test_no_flags_everything_trains(self):
        model = type(self).build()
        for name, param in tunable_named_parameters(model):
            self.assertTrue(param.requires_grad, f"{name} frozen with no flags set")

    def test_freeze_seq_model(self):
        self.check_flags(freeze_seq_model=True)

    def test_freeze_fc_out(self):
        self.check_flags(freeze_fc_out=True)

    def test_freeze_input_pathway(self):
        self.check_flags(freeze_input_pathway=True)

    def test_freeze_all_three(self):
        self.check_flags(freeze_seq_model=True, freeze_fc_out=True,
                         freeze_input_pathway=True)

    def test_seq_plus_input_pathway_leaves_only_the_head(self):
        """The theta+W_in condition: only the output head adapts."""
        model = self.check_flags(freeze_seq_model=True, freeze_input_pathway=True)
        trainable = [n for n, p in tunable_named_parameters(model) if p.requires_grad]
        self.assertTrue(
            trainable and all(n.startswith(('norm_out.', 'fc_out.')) for n in trainable),
            f"expected only the output head trainable, got {sorted(trainable)}",
        )

    def test_frozen_seq_model_still_follows_train_mode(self):
        """A frozen seq_model must STAY in training mode under model.train().

        This looks wrong but is required: cuDNN's fused RNN backward refuses to
        run on a module in eval mode, and gradients still flow through the frozen
        core to reach the trainable input pathway upstream. Forcing it to eval
        raises "cudnn RNN backward can only be called in training mode" on GPU.
        Freezing is enforced by requires_grad, not by module mode.
        """
        model = type(self).build(
            freeze_seq_model=True, freeze_fc_out=True, freeze_input_pathway=True,
        )
        model.train()
        self.assertTrue(
            core_of(model).seq_model.training,
            "frozen seq_model was forced to eval; cuDNN RNN backward will fail on GPU",
        )
        model.eval()
        self.assertFalse(core_of(model).seq_model.training)

    def test_frozen_head_pinned_to_eval(self):
        """The output head carries a real p=0.3 dropout, so it must pin to eval."""
        model = type(self).build(freeze_seq_model=True, freeze_fc_out=True)
        model.train()
        core = core_of(model)
        for name in ('norm_out', 'fc_out', 'out_dropout'):
            self.assertFalse(
                getattr(core, name).training,
                f"{name} back in train mode after .train()",
            )

    @unittest.skipUnless(torch.cuda.is_available(), "needs CUDA for the cuDNN RNN path")
    def test_backward_through_frozen_rnn_core_on_cuda(self):
        """Backward through a frozen LSTM core must work under cuDNN.

        The CPU RNN backward also runs in eval mode, so only a GPU shows this.
        """
        model = type(self).build(freeze_seq_model=True).cuda()
        model.train()
        x, task_index = type(self).make_inputs()
        out, _, _ = model(x.cuda(), task_index.cuda())
        labels = torch.randint(0, OUTPUT_SIZE, (B * T,), device='cuda')
        loss = F.cross_entropy(out.reshape(-1, OUTPUT_SIZE), labels)
        loss.backward()  # raises RuntimeError if seq_model was pinned to eval
        upstream = core_of(model).proj.weight
        self.assertIsNotNone(upstream.grad, "no gradient reached the trainable input pathway")
        self.assertTrue(torch.isfinite(upstream.grad).all())

    def test_frozen_head_dropout_deterministic_without_seq_freeze(self):
        """freeze_fc_out alone should still give a deterministic output head.

        out_dropout sits between norm_out and fc_out, so if it stays in training
        mode the 'frozen' head is not actually deterministic run-to-run.
        """
        model = type(self).build(freeze_fc_out=True)
        model.train()
        self.assertFalse(
            core_of(model).out_dropout.training,
            "out_dropout still in train mode: frozen head is not deterministic",
        )


class TestNaturalFreeze(FreezeFlagMixin, unittest.TestCase):
    """Natural model: CNNRNN over images."""
    build = staticmethod(build_natural)
    make_inputs = staticmethod(natural_inputs)

    def test_vision_backbone_always_frozen(self):
        """The backbone is frozen and in eval regardless of the freeze flags."""
        model = build_natural()
        model.train()
        self.assertFalse(any(p.requires_grad for p in model.vision_backbone.parameters()))
        self.assertFalse(model.vision_backbone.training)


class TestDimMatchFreeze(FreezeFlagMixin, unittest.TestCase):
    """Dimensionality-matched Abstract model: DimMatchCNNRNN wrapping a no-backbone CNNRNN core."""
    build = staticmethod(build_dim_match)
    make_inputs = staticmethod(dim_match_inputs)

    def test_projection_is_a_fixed_buffer(self):
        """The random lift must never be trained, under any flag combination."""
        model = build_dim_match(freeze_seq_model=True)
        self.assertNotIn(
            'projection', dict(model.named_parameters()),
            "projection is a trainable parameter; it must stay a fixed buffer",
        )
        before = model.projection.detach().clone()
        step_once(model, dim_match_inputs())
        self.assertTrue(
            torch.equal(before, model.projection), "projection buffer changed during training",
        )

    def test_noise_makes_forward_stochastic_by_design(self):
        """Sanity check on the control itself: noise is applied every call."""
        model = build_dim_match(noise_scale=0.5, freeze_seq_model=True)
        model.eval()
        x, task_index = dim_match_inputs()
        with torch.no_grad():
            first, _, _ = model(x, task_index)
            second, _, _ = model(x, task_index)
        self.assertFalse(
            torch.equal(first, second),
            "expected fresh per-call noise, got identical outputs",
        )


class TestSubstitutionInputPathway(unittest.TestCase):
    """What the substitution carries across depends on freeze_input_pathway.

    Without it the input pathway is relearned, so every input-pathway tensor
    (norm_in included) must come back fresh. With it the condition is theta+W_in:
    the donor's own input pathway is loaded and frozen, so every tensor must match
    the donor bit for bit.
    """

    def _substitute(self, **freeze_kwargs):
        torch.manual_seed(0)
        donor = core_of(build_dim_match())
        with torch.no_grad():
            for name, p in donor.named_parameters():
                if name.startswith(INPUT_PATHWAY_PREFIXES):
                    p.add_(torch.randn_like(p))  # make every input-pathway tensor non-default
        checkpoint = io.BytesIO()
        torch.save(donor.state_dict(), checkpoint)
        checkpoint.seek(0)
        recipient = core_of(build_dim_match(seq_model_path=checkpoint, **freeze_kwargs))
        return donor.state_dict(), recipient.state_dict()

    def test_core_only_substitution_gets_a_fresh_input_pathway(self):
        d, r = self._substitute(freeze_seq_model=True)
        self.assertTrue(torch.equal(r['norm_in.weight'], torch.ones(HIDDEN)))
        self.assertTrue(torch.equal(r['norm_in.bias'], torch.zeros(HIDDEN)))
        for k in d:
            if k.startswith(INPUT_PATHWAY_PREFIXES):
                self.assertFalse(torch.equal(d[k], r[k]), f'{k} inherited from donor')
            elif k.startswith(SEQ_PREFIXES):
                self.assertTrue(torch.equal(d[k], r[k]), f'{k} not substituted')

    def test_frozen_input_pathway_is_the_donors(self):
        d, r = self._substitute(freeze_seq_model=True, freeze_input_pathway=True)
        for k in d:
            if k.startswith(INPUT_PATHWAY_PREFIXES + SEQ_PREFIXES):
                self.assertTrue(torch.equal(d[k], r[k]), f'{k} not substituted')


class TestModelParity(unittest.TestCase):
    """The two models must expose the same trainable parameters under the same flags."""

    FLAG_SETS = [
        {},
        {'freeze_seq_model': True},
        {'freeze_fc_out': True},
        {'freeze_input_pathway': True},
        {'freeze_seq_model': True, 'freeze_fc_out': True},
        {'freeze_seq_model': True, 'freeze_input_pathway': True},
        {'freeze_seq_model': True, 'freeze_fc_out': True, 'freeze_input_pathway': True},
    ]

    def test_same_trainable_parameters_under_each_flag_set(self):
        for flags in self.FLAG_SETS:
            with self.subTest(flags=flags):
                nat = {n for n, p in tunable_named_parameters(build_natural(**flags))
                       if p.requires_grad}
                dim = {n for n, p in tunable_named_parameters(build_dim_match(**flags))
                       if p.requires_grad}
                self.assertEqual(
                    nat, dim,
                    f"{flags}: models differ. natural-only={sorted(nat - dim)}, "
                    f"dimmatch-only={sorted(dim - nat)}",
                )

    def test_head_only_condition_matches_on_both(self):
        """seq + input pathway frozen must leave exactly the head on both models."""
        flags = {'freeze_seq_model': True, 'freeze_input_pathway': True}
        for label, model in [('natural', build_natural(**flags)),
                             ('dimmatch', build_dim_match(**flags))]:
            with self.subTest(model=label):
                trainable = [n for n, p in tunable_named_parameters(model) if p.requires_grad]
                self.assertTrue(trainable, f"{label}: nothing left to train")
                self.assertTrue(
                    all(n.startswith(('norm_out.', 'fc_out.')) for n in trainable),
                    f"{label}: unexpected trainable params {trainable}",
                )

    def test_everything_frozen_leaves_nothing_trainable(self):
        """All three flags together is degenerate; assert it is, rather than implying use."""
        flags = {'freeze_seq_model': True, 'freeze_fc_out': True,
                 'freeze_input_pathway': True}
        for label, model in [('natural', build_natural(**flags)),
                             ('dimmatch', build_dim_match(**flags))]:
            with self.subTest(model=label):
                trainable = [n for n, p in tunable_named_parameters(model) if p.requires_grad]
                self.assertEqual(trainable, [], f"{label}: expected nothing trainable")


if __name__ == '__main__':
    unittest.main(verbosity=2)
