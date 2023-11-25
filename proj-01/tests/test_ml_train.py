import torch

from src.model import Model4, PositionalEncoding


def test_position_encoding() -> None:
    d_model = 16
    t_shape = (4, 10, d_model)
    pe_layer = PositionalEncoding(d_model)
    x = torch.rand(t_shape)
    assert pe_layer.forward(x).shape == t_shape


def test_padding_mask() -> None:
    torch.manual_seed(2)
    d_model = 16
    batch_size = 2
    seq_length = 5
    item_size = 3
    user_feature_dim = 2
    sample_size = 3
    model = Model4(
        item_size=3,
        user_feature_dim=user_feature_dim,
        num_layers=4,
        d_model=d_model,
        dim_feedforward=d_model,
        nhead=2,
    )
    model.train()
    user_features = torch.rand((batch_size, user_feature_dim))
    item_indices = torch.randint(0, item_size, (batch_size, seq_length))
    target_indices = torch.randint(0, item_size, (batch_size, seq_length, sample_size))
    mask = item_indices == 1
    padding_mask = torch.cat(
        [
            torch.BoolTensor([False for _ in range(batch_size)]).unsqueeze(1),
            item_indices == 2,
        ],
        dim=1,
    )
    target_labels = torch.FloatTensor(
        [
            [[1] + [0] * (sample_size - 1) for _ in range(seq_length)]
            for _ in range(batch_size)
        ]
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for i in range(1000):
        # print("user_features:", user_features)
        # print("item_indices:", item_indices)
        # print("target_indices:", target_indices)
        # print("padding_mask:", padding_mask)
        _, y_target = model.forward(
            user_features, item_indices, target_indices, padding_mask
        )
        # print("y_clv:", y_clv)
        # print("y_target:", y_target)
        # print("target_labels:", target_labels)

        # print("mask:", mask)

        loss = torch.nn.functional.binary_cross_entropy(
            y_target[mask].flatten(), target_labels[mask].flatten()
        )

        # print("y_target:", y_target[mask].flatten())
        # print("target_labels:", target_labels[mask].flatten())
        if i % 100 == 0:
            print("y_target:", y_target[mask])
            print("target_labels:", target_labels[mask])
            print("loss:", loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert False
