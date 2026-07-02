# Transformer Decoder module (paper §3.1).
#
# Hand-written decoder layer built from MultiHeadAttention (attention.py) and
# PositionwiseFeedForward (layers.py). Each layer has three sublayers:
#   1. masked multi-head self-attention (causal mask)
#   2. encoder-decoder (cross) attention over the encoder memory
#   3. position-wise feed-forward
# Post-LayerNorm: LayerNorm(x + Dropout(Sublayer(x))).
import torch
from torch import nn

from Transformer_handmade.config import TransformerConfig
from Transformer_handmade.model.attention import MultiHeadAttention
from Transformer_handmade.model.layers import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """Masked self-attention → cross-attention → feed-forward."""

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(config.d_model, config.h, config.dropout)
        self.cross_attn = MultiHeadAttention(config.d_model, config.h, config.dropout)
        self.feed_forward = PositionwiseFeedForward(
            config.d_model, config.d_ff, config.dropout
        )
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        self.norm3 = nn.LayerNorm(config.d_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

    def forward(
        self,
        x,
        memory,
        tgt_mask=None,
        tgt_key_padding_mask=None,
        memory_key_padding_mask=None,
    ):
        # 1. Masked self-attention sublayer
        sa = self.self_attn(
            x, x, x, key_padding_mask=tgt_key_padding_mask, attn_mask=tgt_mask
        )
        x = self.norm1(x + self.dropout1(sa))
        # 2. Encoder-decoder cross-attention sublayer
        ca = self.cross_attn(
            x, memory, memory, key_padding_mask=memory_key_padding_mask
        )
        x = self.norm2(x + self.dropout2(ca))
        # 3. Feed-forward sublayer
        ff = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff))
        return x


class Seq2SeqDecoder(nn.Module):
    """Stack of N identical decoder layers.

    No extra final LayerNorm: in the paper's Post-LN structure each sublayer
    already ends with LayerNorm, so the last layer's output is normalized.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([DecoderLayer(config) for _ in range(config.N)])

    def forward(self, x, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_mask = self.generate_square_subsequent_mask(x.size(1), x.device)
        for layer in self.layers:
            x = layer(
                x,
                memory,
                tgt_mask=tgt_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return x

    @staticmethod
    def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
        # Boolean mask: True positions are NOT allowed to attend (future tokens).
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool),
            diagonal=1,
        )
