import torch

from src.model import PositionalEncoding


def test_position_encoding() -> None:
    d_model = 16
    t_shape = (4, 10, d_model)
    pe_layer = PositionalEncoding(d_model)
    x = torch.rand(t_shape)
    assert pe_layer.forward(x).shape == t_shape
