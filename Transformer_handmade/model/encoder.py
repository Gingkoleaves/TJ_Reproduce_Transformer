# Transformer Encoder module.
#
# For now this wraps PyTorch's built-in nn.TransformerEncoderLayer stack.
# A later step will replace the internal layer with a hand-written
# combination of multi-head attention + feed-forward blocks.
from torch import nn

from Transformer_handmade.config import TransformerConfig


class Seq2SeqEncoder(nn.Module):
    """Stack of N identical encoder layers (paper §3.1).

    Post-LayerNorm, ReLU feed-forward, batch-first tensors. No extra final
    LayerNorm — the original Transformer applies LayerNorm inside each
    sublayer only.
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.h,
            dim_feedforward=config.d_ff,
            dropout=config.dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.N)

    def forward(self, x, src_key_padding_mask=None):
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)
