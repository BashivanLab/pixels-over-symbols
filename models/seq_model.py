"""
Sequence cores of the task models: vanilla RNN, GRU, LSTM and two causal Transformers.
"""
from typing import Optional, Literal

import torch
import torch.nn as nn

from transformers import BertConfig, BertModel


def init_rnn_hidden(rnn_type, n_layer, batch_size, hidden_size, device):
    """Initialize hidden states for RNN variants for deterministic output"""
    if rnn_type == 'LSTM':
        h_0 = torch.zeros(n_layer, batch_size, hidden_size, device=device).contiguous()
        c_0 = torch.zeros(n_layer, batch_size, hidden_size, device=device).contiguous()
        return h_0, c_0
    else:
        return torch.zeros(n_layer, batch_size, hidden_size, device=device).contiguous()


class RNNSequenceModel(nn.Module):
    """RNN-based sequence model (GRU/LSTM/RNN)."""

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 1,
            rnn_type: Literal['RNN', 'GRU', 'LSTM'] = 'GRU',
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn_type = rnn_type

        rnn_cls = {
            'RNN': nn.RNN,
            'GRU': nn.GRU,
            'LSTM': nn.LSTM
        }[rnn_type]
        self.rnn = rnn_cls(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            for name, param in self.rnn.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
                    if isinstance(self.rnn, nn.LSTM) and 'bias_ih' in name:
                        n = param.size(0)
                        # Set forget gate bias to 1.0
                        param[n // 4:n // 2].fill_(1.0)

    def init_hidden(self, batch_size: int, device: torch.device):
        """Initialize hidden state."""
        return init_rnn_hidden(
            self.rnn_type,
            self.num_layers,
            batch_size,
            self.hidden_size,
            device
        )

    def forward(
            self,
            x: torch.Tensor,
            hidden: Optional[torch.Tensor] = None
    ):
        """
        Args:
            x: (B, T, input_size)
            hidden: Optional hidden state

        Returns:
            output: (B, T, hidden_size)
            hidden: Updated hidden state
        """
        if hidden is None:
            hidden = self.init_hidden(x.size(0), x.device)

        output, hidden = self.rnn(x, hidden)
        return output, hidden


class VanillaDecoderSequenceModel(nn.Module):
    """
    Decoder-only Transformer (the "Transformer" core in the paper): PyTorch's
    nn.TransformerEncoder with pre-norm layers and a causal attention mask, plus
    learned position embeddings.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 4,
            num_heads: int = 8,
            dim_feedforward: int = 2048,
            max_seq_len: int = 6,
            dropout: float = 0.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size

        self.input_proj = nn.Linear(input_size, hidden_size)

        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)

        self._init_weights()
        print(f"[VanillaDecoder] PyTorch TransformerDecoder with {num_layers} layers, {num_heads} heads")

    def _init_weights(self):
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.normal_(self.pos_emb.weight, std=self.hidden_size ** -0.5)

    def forward(
            self,
            x: torch.Tensor,
            hidden=None,
            mask=None,
    ):
        """
        Args:
            x: (B, T, input_size)
            hidden: unused (API compatibility)
            mask: optional padding mask (B, T)

        Returns:
            output: (B, T, hidden_size)
            hidden: None (API compatibility)
        """
        B, T, _ = x.shape
        x = self.input_proj(x)  # (B, T, hidden_size)

        pos = torch.arange(T, device=x.device)
        x = x + self.pos_emb(pos)  # (B, T, hidden_size)

        # Causal mask: decoder sees only past tokens
        causal_mask = nn.Transformer.generate_square_subsequent_mask(T, device=x.device)

        output = self.decoder(
            src=x,
            mask=causal_mask,
            src_key_padding_mask=mask,
            is_causal=True
        )
        return output, None


class BertSequenceModel(nn.Module):
    """
    Randomly initialized Hugging Face BertModel configured as a decoder, so its
    self-attention is causal like the other cores.
    """

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            num_layers: int = 4,
            num_heads: int = 8,
            dim_feedforward: int = 2048,
            max_seq_len: int = 512,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        config = BertConfig(
            vocab_size=1,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            intermediate_size=dim_feedforward,
            max_position_embeddings=max_seq_len,
            hidden_act='gelu',
            layer_norm_eps=1e-12,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            is_decoder=True,
            add_cross_attention=False,
        )
        # input_proj keeps PyTorch's default Linear init (the other cores use Xavier);
        # the trained BERT models were built this way
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.transformer = BertModel(config, add_pooling_layer=False)
        print(f"[Transformer] Using HuggingFace BertModel (pretrained=False) with {num_layers} layers")

    def forward(
            self,
            x: torch.Tensor,
            hidden: Optional[torch.Tensor] = None,
            mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            x: (B, T, input_size)
            mask: Optional attention mask
            hidden: Not used, for API compatibility

        Returns:
            output: (B, T, hidden_size)
            hidden: None (for API compatibility)
        """
        x = self.input_proj(x)  # (B, T, hidden_size)
        output = self.transformer(
            inputs_embeds=x,
            attention_mask=mask,
            return_dict=False
        )[0]

        return output, None


def create_sequence_model(
        model_type: str,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        **kwargs
) -> nn.Module:
    """
    Build a sequence core by name: 'rnn', 'gru', 'lstm', 'decoder' or 'bert' (any case).

    Args:
        model_type: Type of sequence model
        input_size: Input feature dimension
        hidden_size: Hidden state dimension
        num_layers: Number of layers
        **kwargs: Transformer arguments (num_heads, dim_feedforward, max_seq_len)

    Returns:
        Sequence model instance
    """
    if model_type.lower() in ['rnn', 'gru', 'lstm']:
        return RNNSequenceModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            rnn_type=model_type.upper(),
        )
    elif model_type.lower() == 'decoder':
        return VanillaDecoderSequenceModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=kwargs.get('num_heads', 8),
            dim_feedforward=kwargs.get('dim_feedforward', hidden_size * 4),
            max_seq_len=kwargs.get('max_seq_len', 6),
        )
    elif model_type.lower() == 'bert':
        num_heads = kwargs.get('num_heads', 8)
        dim_feedforward = kwargs.get('dim_feedforward', hidden_size * 4)
        max_seq_len = kwargs.get('max_seq_len', 6)

        return BertSequenceModel(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
            max_seq_len=max_seq_len,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
