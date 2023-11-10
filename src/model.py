import torch


class TwoFeedForwardLayer(torch.nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(in_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2.forward(torch.nn.functional.relu(self.fc1.forward(x)))


class Model(torch.nn.Module):
    def __init__(
        self,
        user_embedding_weight: torch.Tensor,
        item_size: int,
        num_layers: int = 4,
        d_model: int = 64,
        dim_feedforward: int = 128,
        nhead: int = 4,
    ) -> None:
        super().__init__()
        user_size, user_feature_dim = user_embedding_weight.shape

        self.user_feature = torch.nn.Embedding(user_size, user_feature_dim)
        self.user_feature.weight = torch.nn.Parameter(user_embedding_weight)
        self.user_feature.weight.requires_grad = False

        self.user_embedding = torch.nn.Linear(user_feature_dim, d_model)

        self.item_embedding = torch.nn.Embedding(item_size, d_model)
        self.item_embedding.weight.requires_grad = False

        self.transformer_layer = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                batch_first=True,
            ),
            num_layers=num_layers,
        )
        self.clv_layer = TwoFeedForwardLayer(d_model, d_model // 2, 1)
        self.target_layer = TwoFeedForwardLayer(d_model, d_model // 2, 1)

    def forward(
        self, user_id: torch.Tensor, item_indices: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        f_u = self.user_feature.forward(user_id)
        e_u = self.user_embedding.forward(f_u)
        H_v = self.item_embedding.forward(item_indices)
        H = torch.cat((e_u.unsqueeze(1), H_v), dim=1)
        H = self.transformer_layer.forward(H)

        y_clv = torch.sigmoid(self.clv_layer.forward(H[:, 0]))
        y_target = torch.sigmoid(self.target_layer.forward(H[:, -1]))

        return y_clv.squeeze(), y_target.squeeze()
