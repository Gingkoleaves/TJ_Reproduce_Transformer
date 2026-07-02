# Transformer Decoder module.
#
# For now this wraps PyTorch's built-in nn.TransformerDecoderLayer stack.
# A later step will replace the internal layer with a hand-written
# combination of masked multi-head attention + cross-attention + feed-forward.
import torch
from torch import nn

from Transformer_handmade.config import TransformerConfig


class Seq2SeqDecoder(nn.Module):
    """Stack of N identical decoder layers (paper §3.1).

    Each layer: masked self-attention → encoder-decoder cross-attention →
    feed-forward. Post-LayerNorm, ReLU, batch-first. No extra final LayerNorm.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.h,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=config.N)

    def forward(self, x, memory, tgt_key_padding_mask=None, memory_key_padding_mask=None):
        tgt_mask = self.generate_square_subsequent_mask(x.size(1), x.device)
        return self.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )

    @staticmethod
    def generate_square_subsequent_mask(size: int, device: torch.device) -> torch.Tensor:
        # Boolean mask: True positions are NOT allowed to attend (future tokens).
        return torch.triu(
            torch.ones(size, size, device=device, dtype=torch.bool),
            diagonal=1,
        )
