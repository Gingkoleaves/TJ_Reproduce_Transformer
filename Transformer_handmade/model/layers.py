# Transformer basic layers (paper §3.3).
#
# Position-wise feed-forward network, applied identically and independently
# to each position:  FFN(x) = max(0, x W_1 + b_1) W_2 + b_2.
from torch import nn


class PositionwiseFeedForward(nn.Module):
    """Two linear transformations with a ReLU in between (paper §3.3).

    Inner dimensionality is d_ff = 2048 for the base model.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
