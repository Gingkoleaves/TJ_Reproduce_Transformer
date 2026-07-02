# Multi-head attention module for the Transformer (paper §3.2).
#
# Implements Scaled Dot-Product Attention and the multi-head wrapper.
# Supports two mask kinds:
#   - key_padding_mask: (B, T_k) bool, True marks padding key positions.
#   - attn_mask:        (T_q, T_k) bool, True marks positions that must NOT
#                       be attended to (used for the decoder causal mask).
import math

import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    """MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O.

    Each head runs scaled dot-product attention on d_k = d_model / h
    dimensional projections (paper §3.2.2, d_k = d_v = 64 for the base model).
    """

    def __init__(self, d_model: int, h: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % h == 0, f"d_model ({d_model}) must be divisible by h ({h})"
        self.d_model = d_model
        self.h = h
        self.d_k = d_model // h

        # One projection each for Q, K, V, and the output — all d_model → d_model.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, T, d_model) -> (B, h, T, d_k)
        b, t, _ = x.shape
        return x.view(b, t, self.h, self.d_k).transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        # (B, h, T, d_k) -> (B, T, d_model)
        b, _, t, _ = x.shape
        return x.transpose(1, 2).contiguous().view(b, t, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        q = self._split_heads(self.w_q(query))  # (B, h, T_q, d_k)
        k = self._split_heads(self.w_k(key))    # (B, h, T_k, d_k)
        v = self._split_heads(self.w_v(value))  # (B, h, T_k, d_k)

        # Scaled dot-product scores: (B, h, T_q, T_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        if attn_mask is not None:
            # (T_q, T_k) broadcasts over batch and heads.
            scores = scores.masked_fill(attn_mask, float("-inf"))
        if key_padding_mask is not None:
            # (B, T_k) -> (B, 1, 1, T_k)
            scores = scores.masked_fill(
                key_padding_mask[:, None, None, :], float("-inf")
            )

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        context = torch.matmul(attn, v)          # (B, h, T_q, d_k)
        return self.w_o(self._merge_heads(context))
