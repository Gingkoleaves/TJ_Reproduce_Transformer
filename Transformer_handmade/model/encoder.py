# Transformer Encoder module (paper §3.1).
#
# Hand-written encoder layer built from MultiHeadAttention (attention.py) and
# PositionwiseFeedForward (layers.py). Post-LayerNorm: each sublayer computes
# LayerNorm(x + Dropout(Sublayer(x))).
from torch import nn

from Transformer_handmade.config import TransformerConfig
from Transformer_handmade.model.attention import MultiHeadAttention
from Transformer_handmade.model.layers import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """Self-attention sublayer followed by a feed-forward sublayer."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(config.d_model, config.h, config.dropout)
        self.feed_forward = PositionwiseFeedForward(
            config.d_model, config.d_ff, config.dropout
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

    def forward(self, x, src_key_padding_mask=None):
        # Self-attention sublayer
        attn = self.self_attn(x, x, x, key_padding_mask=src_key_padding_mask)
        x = self.norm1(x + self.dropout1(attn))
        # Feed-forward sublayer
        ff = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff))
        return x


class Seq2SeqEncoder(nn.Module):
    """Stack of N identical encoder layers with a final LayerNorm.

    Matches ``nn.Transformer``, which applies a LayerNorm to the encoder
    output. This bounds the residual-stream magnitude feeding the decoder's
    cross-attention.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.N)])
        self.norm = nn.LayerNorm(config.d_model)

    def forward(self, x, src_key_padding_mask=None):
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(x)
