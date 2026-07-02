# Embedding module for Transformer, positional encoding and combine with token embedding.
# The embedding layer is responsible for converting the input tokens into dense vectors
# that can be processed by the rest of the model.
# It also adds positional encoding to the token embeddings to provide information about
# the position of each token in the sequence.
import math

import torch
from torch import nn


class Seq2SeqEmbedding(nn.Module):
    """Token embeddings + sinusoidal positional encoding + output generator.

    Follows "Attention Is All You Need" §3.4:
    - Embedding weights are multiplied by sqrt(d_model).
    - Input embedding, output embedding, and the pre-softmax linear share
      the same weight matrix (weight tying) when ``share_embeddings`` is set.
    """

    def __init__(
        self,
        config,
        src_vocab_size: int,
        tgt_vocab_size: int,
        share_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.scale = math.sqrt(config.d_model)

        self.src_embedding = nn.Embedding(src_vocab_size, config.d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.d_model, config.dropout, config.max_seq_len
        )
        self.generator = nn.Linear(config.d_model, tgt_vocab_size)

        if share_embeddings:
            self.share_weights(src_vocab_size, tgt_vocab_size)

    def share_weights(self, src_vocab_size: int, tgt_vocab_size: int) -> None:
        assert src_vocab_size == tgt_vocab_size, (
            f"Shared embeddings require src_vocab_size == tgt_vocab_size, "
            f"got {src_vocab_size} != {tgt_vocab_size}"
        )
        self.tgt_embedding.weight = self.src_embedding.weight
        self.generator.weight = self.src_embedding.weight

    def embed_src(self, src: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding(self.src_embedding(src) * self.scale)

    def embed_tgt(self, tgt: torch.Tensor) -> torch.Tensor:
        return self.positional_encoding(self.tgt_embedding(tgt) * self.scale)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
