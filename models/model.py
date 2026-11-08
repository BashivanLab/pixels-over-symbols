"""Task models: a sequence core with an optional vision frontend.

- :class:`CustomRNN` (the one-hot Abstract model) consumes per-frame stimulus
  embeddings concatenated with a task one-hot, and runs them through a sequence
  core (RNN/GRU/LSTM/Transformer/BERT) to a small action head.
- :class:`CNNRNN` (the Natural model) adds a frozen `timm` vision backbone plus
  attention pooling in front of the same sequence core, so it can take image
  frames or their cached backbone features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.seq_model import create_sequence_model
from models.vision_model import load_vision_backbone
from utils.helper import load_repair_checkpoint, remove_prefix


def make_divisible(v: int, divisor: int) -> int:
    """Rounds v up to the nearest multiple of divisor."""
    return ((v + divisor - 1) // divisor) * divisor


def gaussian_noise_like(x, seeds=None):
    """Standard normal noise shaped like ``x``.

    ``seeds=None`` draws from the global RNG (training: fresh noise every call).
    Otherwise ``seeds`` holds one int per sample along dim 0, and sample ``i`` is
    drawn from its own generator seeded with ``seeds[i]``, so a trial's noise does
    not depend on batch composition or on what ran before (seeded evaluation).
    """
    if seeds is None:
        return torch.randn_like(x)
    assert len(seeds) == x.shape[0], f"{len(seeds)} noise seeds for a batch of {x.shape[0]}"
    noise = torch.empty_like(x)
    g = torch.Generator(device=x.device)
    for i, s in enumerate(seeds):
        noise[i].normal_(generator=g.manual_seed(int(s)))
    return noise


def check_trainable_params(model):
    """Print each parameter's name/shape, marking frozen ones, and the total trainable count."""
    print("\n--- Trainable Parameters ---")
    total_trainable = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"{name:50} | Shape: {str(param.shape):20}")
            total_trainable += param.numel()
        else:
            print(f"{name:50} | Frozen")
    print(f"Total Trainable Params: {total_trainable:,}")
    print("---------------------------\n")
    return


class AttentionPoolGeneric(nn.Module):
    """CLIP-style attention pooling over a token sequence.

    Prepends a learned mean-token query and attends over all input tokens to
    produce a single ``output_dim`` vector per item, letting the model learn
    which spatial/patch tokens matter rather than using flat average pooling.
    """

    def __init__(
            self,
            num_input_tokens: int,
            embed_dim: int,
            num_heads: int,
            output_dim: int = None
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, ''
        # Initialize positional embedding with small variance
        self.positional_embedding = nn.Parameter(
            torch.randn(num_input_tokens + 1, embed_dim) / embed_dim ** 0.5
        )
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
        self.num_heads = num_heads

        self._init_weights()

    def _init_weights(self):
        # Xavier initialization is standard for Attention projections
        nn.init.xavier_uniform_(self.k_proj.weight)
        nn.init.xavier_uniform_(self.q_proj.weight)
        nn.init.xavier_uniform_(self.v_proj.weight)
        nn.init.xavier_uniform_(self.c_proj.weight)

        # Zero out biases
        for m in [self.k_proj, self.q_proj, self.v_proj, self.c_proj]:
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x shape: [L, BT, D]
        # Safety check for resolution changes
        L, BT, D = x.shape
        assert L + 1 == self.positional_embedding.shape[0]

        # Prepend mean token
        x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # [L+1, BT, D]

        # Add Positional Embedding
        x = x + self.positional_embedding[:, None, :].to(x.dtype)

        # Multi-head Attention
        x, _ = F.multi_head_attention_forward(
            query=x[:1], key=x, value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False
        )
        return x.squeeze(0)  # [BT, D]


class CustomRNN(nn.Module):
    """Sequence model over pre-computed embeddings for the abstract task.

    Projects ``[embedding | task_one_hot]`` into ``hidden_size``, runs it through
    the configured sequence core, and decodes a per-timestep action. Optionally
    loads a pretrained sequence core (``seq_model_path``) and freezes the core
    and/or output head for fine-tuning. ``hidden_size`` is rounded up to a
    multiple of ``num_heads`` for the attention/Transformer variants.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            dropout: float = 0.3,
            output_size: int = 2,
            label_size: int = 43,
            seq_model_type: str = 'GRU',
            num_seq_layers: int = 1,
            seq_model_path: str = None,
            freeze_seq_model: bool = False,
            freeze_fc_out: bool = False,
            # Transformer-specific args
            num_heads: int = 8,
            dim_feedforward: int = None,
    ):
        super().__init__()
        hidden_size = make_divisible(hidden_size, num_heads)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.label_size = label_size
        self.seq_model_type = seq_model_type.lower()
        self.freeze_fc_out = freeze_fc_out
        self.freeze_seq_model = freeze_seq_model

        if dim_feedforward is None:
            dim_feedforward = hidden_size * 4
        self.fc_in = nn.Linear(input_size, hidden_size)
        self.norm_in = nn.LayerNorm(hidden_size)

        self.seq_model = create_sequence_model(
            model_type=seq_model_type,
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_seq_layers,
            # Transformer args
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            max_seq_len=6,
        )

        # Output Head
        self.norm_out = nn.LayerNorm(hidden_size)
        self.out_dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

        # Initialize Weights
        self._init_weights()

        if seq_model_path is not None:
            self._load_custom_rnn_weights(seq_model_path)
            # reinit the finetuning weights
            nn.init.xavier_uniform_(self.fc_in.weight)
            nn.init.constant_(self.fc_in.bias, 0)
            if not self.freeze_fc_out:
                nn.init.xavier_uniform_(self.fc_out.weight)
                nn.init.constant_(self.fc_out.bias, 0)

        if freeze_seq_model:
            self.freeze_components([self.seq_model])
        if freeze_fc_out:
            self.freeze_components([self.norm_out, self.fc_out])

        if freeze_fc_out or freeze_seq_model:
            check_trainable_params(self)

    def train(self, mode: bool = True):
        super().train(mode)
        # A frozen seq_model deliberately still FOLLOWS `mode` (i.e. it is left in
        # training mode here). Do not "fix" this by forcing it to eval: cuDNN's
        # fused RNN backward refuses to run on a module in eval mode, and
        # gradients still flow *through* the frozen core to reach the trainable
        # input pathway upstream, so eval raises
        #   RuntimeError: cudnn RNN backward can only be called in training mode
        # Freezing is enforced by requires_grad, not by module mode, and every
        # core is built with internal dropout=0, so following `mode` is inert.
        #
        # The output head is different: out_dropout is a real p=0.3 dropout, so a
        # frozen head is pinned to eval to be deterministic, whether or not the
        # sequence core is frozen too.
        if self.freeze_fc_out:
            self.norm_out.eval()
            self.fc_out.eval()
            self.out_dropout.eval()
        return self

    def _load_custom_rnn_weights(self, path):
        """Loads sequence-core weights from a CustomRNN, CNNRNN or DimMatchCNNRNN checkpoint.

        Donor checkpoints carry different key prefixes depending on how they were
        trained (see :meth:`CNNRNN._load_custom_rnn_weights`, which this mirrors):
        ``_orig_mod.`` for a ``torch.compile``-wrapped donor (already stripped by
        :func:`load_repair_checkpoint`), and ``core.`` for a ``DimMatchCNNRNN``
        donor, which holds its CNNRNN as ``self.core``.
        """
        state_dict = load_repair_checkpoint(path, device='cpu')
        state_dict = {remove_prefix(k, 'core.'): v for k, v in state_dict.items()}

        keys_to_ignore = ['fc_in.weight', 'fc_in.bias']
        state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_ignore}
        # --- 1. Validate Hidden Size & Output Size ---
        if 'fc_out.weight' in state_dict:
            loaded_out, loaded_hidden = state_dict['fc_out.weight'].shape
            assert loaded_hidden == self.hidden_size, \
                f"Hidden size mismatch! Loaded CustomRNN had hidden_size={loaded_hidden}, but CNNRNN expects {self.hidden_size}."
            assert loaded_out == self.output_size, \
                f"Output size mismatch! Loaded CustomRNN had output_size={loaded_out}, but CNNRNN expects {self.output_size}."

        state_dict = {k: v for k, v in state_dict.items() if 'fc_in' not in k}
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        target_prefixes = ('seq_model.', 'norm_out.', 'fc_out.')
        missing_critical_keys = [k for k in missing if k.startswith(target_prefixes)]

        assert len(missing_critical_keys) == 0, \
            f"CRITICAL FAILURE: Failed to load some sequence components! Missing keys: {missing_critical_keys}"

        print(f"Successfully loaded and verified seq model weights from {path}.")
        return

    def freeze_components(self, components_to_freeze):
        """Freezes the sequence model and output heads."""
        for component in components_to_freeze:
            for param in component.parameters():
                param.requires_grad = False
            # Ensure dropout/batchnorm act deterministically
            component.eval()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.constant_(self.fc_in.bias, 0)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)

    def forward(self, x, task_index, actions=None, get_act=False):
        # x has shape [B, T, input_size]
        task_features = task_index.unsqueeze(1).expand(-1, x.shape[1], -1).float()
        x = torch.cat([x, task_features], dim=-1)  # concatenate the task embedding
        x = self.fc_in(x)
        x = self.norm_in(x)
        x = F.gelu(x)

        pre_seq = x
        seq_out, _ = self.seq_model(x)

        seq_norm = self.norm_out(seq_out)
        seq_out_dropped = self.out_dropout(seq_norm)
        out = self.fc_out(seq_out_dropped)  # Take the output of the last time step

        if get_act:
            hidden_states = []
            pre_states = []

            for t in range(out.size(1)):
                hidden_states.append(seq_out[:, t, :].detach().cpu().numpy())
                pre_states.append(pre_seq[:, t, :].detach().cpu().numpy())

            return out, actions, {
                "pre": pre_states,  # fc_in trajectories
                "post": hidden_states  # seq_model trajectories
            }
        return out, actions, None


class CNNRNN(nn.Module):
    """Vision-frontend variant of :class:`CustomRNN` for the naturalistic task.

    A frozen `timm` backbone (always kept in eval, no gradient) encodes each
    frame; ResNet feature maps and ViT token grids are both reduced to a token
    sequence, projected, and attention-pooled to one vector per frame. That
    vector is concatenated with the task one-hot and fed to the same sequence
    core + action head as :class:`CustomRNN`. The ``forward`` also accepts
    pre-computed 4-D embeddings to bypass the backbone.
    """

    def __init__(
            self,
            hidden_size: int,
            dropout: float = 0.3,
            output_size: int = 2,
            label_size: int = 43,
            cnn_type: str = 'resnet50',
            seq_model_type: str = 'GRU',
            num_seq_layers: int = 1,
            pretrained: bool = True,
            # Random-encoder model: pins the exact random backbone draw, so the
            # encoder that produced the feature cache is the one in the checkpoint.
            vision_state_dict_path: str = None,
            seq_model_path: str = None,
            freeze_seq_model: bool = False,
            freeze_fc_out: bool = False,
            # Freezes the whole input pathway: proj/attn_pool/fc_in/norm_in.
            # With seq_model_path the pathway is LOADED from the donor (not
            # re-initialized), so this is the theta+W_in substitution: donor
            # core and donor input projection frozen, only the readout retrains.
            freeze_input_pathway: bool = False,
            # Transformer-specific args
            num_heads: int = None,
            dim_feedforward: int = None,
            # Synthetic-token / no-backbone mode (dimensionality-matched model):
            # when set, the vision backbone is skipped and proj/attn_pool are sized
            # from feat_dim/num_input_tokens, so the model consumes pre-computed
            # (B, T, num_input_tokens, feat_dim) token tensors directly.
            no_backbone: bool = False,
            feat_dim: int = None,
            num_input_tokens: int = None,
    ):
        super().__init__()
        if num_heads is None:
            # For attention pool 2d
            num_heads = hidden_size // 32
        hidden_size = make_divisible(hidden_size, num_heads)

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.label_size = label_size
        self.seq_model_type = seq_model_type.lower()
        self.freeze_fc_out = freeze_fc_out
        self.freeze_seq_model = freeze_seq_model
        self.freeze_input_pathway = freeze_input_pathway

        # 1. Vision backbone (skipped when consuming pre-computed synthetic tokens)
        if no_backbone:
            assert feat_dim is not None and num_input_tokens is not None, \
                "no_backbone=True requires feat_dim and num_input_tokens"
            self.vision_backbone = None
            self.backbone_is_vit = False
            backbone_out_dim = feat_dim
            num_tokens = num_input_tokens
        else:
            self.vision_backbone = load_vision_backbone(
                cnn_type=cnn_type,
                pretrained=pretrained,
                state_dict_path=vision_state_dict_path
            )
            for param in self.vision_backbone.parameters():
                param.requires_grad = False
            self.vision_backbone.eval()

            # Determine backbone output shapes
            with torch.no_grad():
                dummy_feat = self.vision_backbone.forward_features(torch.randn(1, 3, 224, 224))
                self.backbone_is_vit = len(dummy_feat.shape) == 3

                if self.backbone_is_vit:
                    backbone_out_dim = dummy_feat.shape[-1]
                    num_tokens = dummy_feat.shape[1]  # size patch * patch
                else:
                    backbone_out_dim = dummy_feat.shape[1]
                    num_tokens = dummy_feat.shape[2] * dummy_feat.shape[3]  # size patch * patch

        # 2. Vision Projection & Pooling
        self.proj = nn.Linear(backbone_out_dim, hidden_size)
        self.attn_pool = AttentionPoolGeneric(
            num_input_tokens=num_tokens,
            embed_dim=hidden_size,
            num_heads=num_heads,
            output_dim=hidden_size
        )

        # 3. Task one-hot is concatenated to the pooled frame vector
        seq_input_size = hidden_size + label_size

        self.fc_in = nn.Linear(seq_input_size, hidden_size)
        self.norm_in = nn.LayerNorm(hidden_size)
        self.input_dropout = nn.Dropout(0.1)

        if dim_feedforward is None:
            dim_feedforward = hidden_size * 4
        # 4. Sequence Model
        self.seq_model = create_sequence_model(
            model_type=seq_model_type,
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_seq_layers,
            # Transformer args
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            max_seq_len=6,
        )

        # 5. Output Head
        self.norm_out = nn.LayerNorm(hidden_size)
        self.out_dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, output_size)

        # Initialize Weights
        self._init_weights()

        if seq_model_path is not None:
            self._load_custom_rnn_weights(
                seq_model_path, load_input_pathway=freeze_input_pathway)
            if not freeze_input_pathway:
                # The input pathway is relearned for the new token content, so start
                # it from the same place for every donor type: a CustomRNN donor has
                # no proj/attn_pool keys to load, but a CNNRNN or DimMatchCNNRNN donor
                # does, and letting it warm-start would make donor type change the
                # initialization rather than only the substituted core.
                nn.init.xavier_uniform_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)
                nn.init.xavier_uniform_(self.fc_in.weight)
                nn.init.constant_(self.fc_in.bias, 0)
                self.norm_in.reset_parameters()
                self.attn_pool._init_weights()
                nn.init.normal_(
                    self.attn_pool.positional_embedding, mean=0.0, std=self.hidden_size ** -0.5
                )
            if not self.freeze_fc_out:
                nn.init.xavier_uniform_(self.fc_out.weight)
                nn.init.constant_(self.fc_out.bias, 0)

        if freeze_seq_model:
            self.freeze_components([self.seq_model])
        if freeze_fc_out:
            self.freeze_components([self.norm_out, self.fc_out])
        if freeze_input_pathway:
            # Everything upstream of the sequence core, frozen at the donor's
            # weights (loaded above; without a donor, at random init). Combined
            # with freeze_seq_model it leaves norm_out/fc_out as the only
            # trainable module (theta+W_in).
            # input_dropout is included for the same reason freeze_fc_out pins
            # out_dropout: it is the last step of the input pathway before the
            # core, so leaving p=0.1 active would mean the "frozen" input
            # pathway still injects fresh noise every step.
            self.freeze_components([self.proj, self.attn_pool, self.fc_in,
                                    self.norm_in, self.input_dropout])

        if freeze_fc_out or freeze_seq_model or freeze_input_pathway:
            check_trainable_params(self)

    def _load_custom_rnn_weights(self, path, load_input_pathway=False):
        """Loads sequence-core weights from a CustomRNN, CNNRNN or DimMatchCNNRNN checkpoint.

        Donor checkpoints carry different key prefixes depending on how they were
        trained, so both are stripped before matching:

        - ``_orig_mod.`` when the donor was wrapped by ``torch.compile`` (GPU runs;
          the CPU-trained abstract models have no prefix).
        - ``core.`` when the donor was a ``DimMatchCNNRNN``, which holds its CNNRNN
          as ``self.core``.

        The input pathway (``proj``/``attn_pool``/``fc_in``/``norm_in``) is dropped
        unless ``load_input_pathway``: by default the constructor re-initializes it
        right after this load (so it is relearned for the new token content), which
        makes loading it pointless and, for a donor whose input geometry differs
        from the recipient (e.g. a ResNet-shaped DimMatch donor into a ViT
        recipient, 2048->768 channels and 50->198 tokens), a fatal size mismatch
        instead of a silent overwrite. With ``load_input_pathway`` the donor's
        pathway is required: it must exist (a CustomRNN donor has none) and match
        the recipient's geometry.
        """
        state_dict = load_repair_checkpoint(path, device='cpu')
        state_dict = state_dict.get('state_dict', state_dict) if isinstance(state_dict, dict) else state_dict
        state_dict = {remove_prefix(k, 'core.'): v for k, v in state_dict.items()}
        input_pathway_prefixes = ('proj.', 'attn_pool.', 'fc_in.', 'norm_in.')
        if not load_input_pathway:
            state_dict = {k: v for k, v in state_dict.items()
                          if not k.startswith(input_pathway_prefixes)}

        # --- 1. Validate Hidden Size & Output Size ---
        if 'fc_out.weight' in state_dict:
            loaded_out, loaded_hidden = state_dict['fc_out.weight'].shape
            assert loaded_hidden == self.hidden_size, \
                f"Hidden size mismatch! Loaded CustomRNN had hidden_size={loaded_hidden}, but CNNRNN expects {self.hidden_size}."
            assert loaded_out == self.output_size, \
                f"Output size mismatch! Loaded CustomRNN had output_size={loaded_out}, but CNNRNN expects {self.output_size}."

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        target_prefixes = ('seq_model.', 'norm_out.', 'fc_out.')
        if load_input_pathway:
            target_prefixes += input_pathway_prefixes
        missing_critical_keys = [k for k in missing if k.startswith(target_prefixes)]

        assert len(missing_critical_keys) == 0, \
            f"CRITICAL FAILURE: Failed to load some sequence components! Missing keys: {missing_critical_keys}"

        print(f"Successfully loaded and verified seq model weights from {path}.")
        return

    def freeze_components(self, components_to_freeze):
        """Freezes the sequence model and output heads."""
        for component in components_to_freeze:
            for param in component.parameters():
                param.requires_grad = False
            # Ensure dropout/batchnorm act deterministically
            component.eval()

    def _init_weights(self):
        """Xavier init with zero bias for the vision projection and the input/output layers."""
        # 1. Vision Projector
        nn.init.xavier_uniform_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

        # 2. Input/Output MLPs
        nn.init.xavier_uniform_(self.fc_in.weight)
        nn.init.constant_(self.fc_in.bias, 0)
        nn.init.xavier_uniform_(self.fc_out.weight)
        nn.init.constant_(self.fc_out.bias, 0)

    def train(self, mode: bool = True):
        super().train(mode)
        # Always keep backbone frozen and in eval (no BN updates)
        if self.vision_backbone is not None:
            self.vision_backbone.eval()

        # A frozen seq_model deliberately still FOLLOWS `mode` (i.e. it is left in
        # training mode here). Do not "fix" this by forcing it to eval: cuDNN's
        # fused RNN backward refuses to run on a module in eval mode, and
        # gradients still flow *through* the frozen core to reach the trainable
        # proj/attn_pool/fc_in upstream, so eval raises
        #   RuntimeError: cudnn RNN backward can only be called in training mode
        # Freezing is enforced by requires_grad, not by module mode, and every
        # core is built with internal dropout=0, so following `mode` is inert.
        #
        # The output head is different: out_dropout is a real p=0.3 dropout, so a
        # frozen head is pinned to eval to be deterministic, whether or not the
        # sequence core is frozen too.
        if self.freeze_fc_out:
            self.norm_out.eval()
            self.fc_out.eval()
            self.out_dropout.eval()
        if self.freeze_input_pathway:
            # proj/fc_in/norm_in/attn_pool are all mode-inert (Linear, LayerNorm,
            # attention with dropout_p=0); input_dropout is not, and it is the
            # reason this block exists.
            self.proj.eval()
            self.attn_pool.eval()
            self.fc_in.eval()
            self.norm_in.eval()
            self.input_dropout.eval()
        return self

    def forward(self, input_img, task_index, actions=None, hidden_state=None, get_act=False):
        if len(input_img.shape) == 5:  # images
            B, T, C, H, W = input_img.shape
            frames = input_img.reshape(B * T, C, H, W)  # combine batch and length for speed-up

            with torch.no_grad():
                features = self.vision_backbone.forward_features(frames)

            if not self.backbone_is_vit:
                # ResNet: (BT, C, H, W) -> (BT, C, HW) -> (BT, HW, C)
                # we want size patch * patch
                features = features.flatten(2).transpose(1, 2)
        elif len(input_img.shape) == 4:  # precomputed image embeddings
            B, T, L, D = input_img.shape
            features = input_img.reshape(B * T, L, D)
        else:
            raise RuntimeError(f'input shape length should be 4 or 5, found {len(input_img.shape)}')
        cnn_features = features.detach().clone() if get_act else None

        # --- Projection & Pooling ---
        features = self.proj(features)  # (BT, L, D)
        features = features.permute(1, 0, 2)  # (L, BT, D) for AttnPool

        pooled_feats = self.attn_pool(features)  # (BT, D)
        flattened = pooled_feats.view(B, T, self.hidden_size)

        # --- Task Context ---
        task_index = task_index.float()
        task_info = task_index.unsqueeze(1).expand(-1, T, -1)

        x = torch.cat([flattened, task_info], dim=-1)
        x = self.fc_in(x)
        x = self.norm_in(x)
        x = F.gelu(x)
        x = self.input_dropout(x)

        pre_seq = x
        seq_out, _ = self.seq_model(x, hidden_state)

        seq_out_norm = self.norm_out(seq_out)
        seq_out_dropped = self.out_dropout(seq_out_norm)
        out = self.fc_out(seq_out_dropped)

        if get_act:
            hidden_states = []
            pre_states = []

            for t in range(T):
                hidden_states.append(seq_out[:, t].detach().cpu().numpy())
                pre_states.append(pre_seq[:, t].detach().cpu().numpy())

            return out, actions, {
                "cnn": cnn_features,
                "pre": pre_states,
                "post": hidden_states
            }
        return out, actions, None
