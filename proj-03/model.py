import torch


class Model(torch.nn.Module):
    def __init__(self, d_model: int, user_n: int, item_n: int):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=user_n, embedding_dim=d_model
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=item_n, embedding_dim=d_model
        )
        self.linears = torch.nn.Sequential(
            torch.nn.Linear(2 * d_model, 2 * d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, 1),
            torch.nn.Sigmoid(),
        )

    def forward(
        self,
        u: torch.LongTensor,
        i: torch.LongTensor,
    ) -> torch.FloatTensor:
        eu = self.user_embedding.forward(u)
        ei = self.item_embedding.forward(i)
        x = torch.cat([eu, ei], dim=-1)
        y = self.linears.forward(x)
        return y.squeeze()


class MatrixFactorization(torch.nn.Module):
    def __init__(self, user_n: int, item_n: int, d_model: int):
        super().__init__()
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=user_n, embedding_dim=d_model
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=item_n, embedding_dim=d_model
        )
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, u: torch.LongTensor, i: torch.LongTensor) -> torch.Tensor:
        e_u = self.user_embedding.forward(u)
        e_i = self.item_embedding.forward(i)
        return self.sigmoid.forward((e_u * e_i).sum(-1))
