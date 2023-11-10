import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.model import *


@dataclass
class Args:
    d_model: int = 32
    batch_size: int = 64
    nhead: int = 4
    epochs: int = 10
    lr: float = 1e-3
    sample_size: int = 5
    alpha: float = 0.5
    context_item_size: int = 9
    device: str = "cpu"
    model_path: str = "model.model"


class Dataset(torch.utils.data.Dataset):
    """
    self.features = [(user_id, [context_item_id])]
    self.target_items = [[target_item]]
    self.clv = [clv_value]
    """

    def __init__(
        self,
        sequences: dict[int, list[int]],
        clv_dict: dict[int, float],
        target_items: dict[int, list[int]],
    ):
        """
        Args:
            sequences (dict[int, list[int]]):
                ユーザごとの購買商品系列
                [user_id : user_items]
            clv_dict (dict[int, float]):
                ユーザのCLV
                [user_id : clv_value]
            target_items (dict[int, list[int]]):
                ユーザの将来CV商品
                [user_id : target_items]
        """
        self.features = []
        self.target_items = []
        self.clv = []

        for user_idx, sequence in tqdm(sequences.items()):
            self.features.append((user_idx, sequence))
            self.target_items.append(
                target_items[user_idx] if user_idx in target_items else []
            )
            self.clv.append(clv_dict[user_idx] if user_idx in clv_dict else 0)

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[tuple[int, list[int]], list[int], float]:
        return self.features[idx], self.target_items[idx], self.clv[idx]


def load_interaction_df(
    last_review_date: pd.Timestamp | str,
    train_split_date: pd.Timestamp | str,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    interaction_df = pd.read_csv("../data/ml-25m/ratings.csv").rename(
        columns={"userId": "user_id", "movieId": "item_id"}
    )
    interaction_df.timestamp = pd.to_datetime(interaction_df.timestamp, unit="s")

    first_review = interaction_df.groupby("user_id").timestamp.min()
    last_review = interaction_df.groupby("user_id").timestamp.max()
    target_users = set(last_review[(first_review < train_split_date)].index) & set(
        last_review[(last_review_date <= last_review)].index
    )

    interaction_df = interaction_df[
        interaction_df.user_id.isin(target_users)
    ].reset_index(drop=True)

    interaction_df, user_le, item_le = encode_user_item_id(
        interaction_df=interaction_df
    )
    return interaction_df, user_le, item_le


def create_user_features(
    feature_df: pd.DataFrame, split_date: pd.Timestamp | str
) -> pd.DataFrame:
    # 最新購買日
    recency = split_date - feature_df.groupby("user_id").timestamp.max()  # type: ignore
    recency = (recency.dt.days / 365).rename("recency")
    recency[recency >= 1] = 1

    # 総購買数
    frequency = feature_df.user_id.value_counts().sort_index().rename("frequency")
    frequency /= 500
    frequency[frequency >= 1] = 1

    # 利用期間
    tenure = (
        feature_df.groupby("user_id").timestamp.max()
        - feature_df.groupby("user_id").timestamp.min()
    )
    tenure = (tenure.dt.days / 365).rename("tenure")
    tenure[tenure >= 1] = 1

    user_features = pd.merge(recency, tenure, on="user_id", how="left")
    user_features = pd.merge(user_features, frequency, on="user_id", how="left")
    return user_features


def create_targets(target_df: pd.DataFrame) -> tuple[dict, dict]:
    # CLV
    clv = target_df.user_id.value_counts().sort_index()
    threshold = clv.quantile(0.95)
    clv /= threshold
    clv[clv >= 1] = 1
    clv_dict = clv.to_dict()

    # CV商品
    target_items = target_df.groupby("user_id").item_id.agg(list).to_dict()

    return clv_dict, target_items


def encode_user_item_id(
    interaction_df: pd.DataFrame,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    user_le = LabelEncoder().fit(interaction_df.user_id)
    item_le = LabelEncoder().fit(interaction_df.item_id)

    interaction_df.user_id = user_le.transform(interaction_df.user_id)
    interaction_df.item_id = item_le.transform(interaction_df.item_id)

    return interaction_df, user_le, item_le


def create_dataset(
    interaction_df: pd.DataFrame,
    train_split_date: pd.Timestamp | str,
    test_split_date: pd.Timestamp | str,
) -> tuple[Dataset, Dataset, torch.Tensor]:
    feature_df = (
        interaction_df[interaction_df.timestamp < train_split_date]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    target_df = (
        interaction_df[
            (interaction_df.timestamp >= train_split_date)
            & (interaction_df.timestamp < test_split_date)
        ]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    test_df = (
        interaction_df[interaction_df.timestamp >= test_split_date]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    sequences = feature_df.groupby("user_id").item_id.agg(list).to_dict()
    user_features = create_user_features(
        feature_df=feature_df, split_date=train_split_date
    )
    user_embedding_weight = torch.from_numpy(user_features.values.astype(np.float32))

    train_clv_dict, train_target_items = create_targets(target_df=target_df)
    test_clv_dict, test_target_items = create_targets(target_df=test_df)

    train_dataset = Dataset(
        sequences=sequences,
        clv_dict=train_clv_dict,
        target_items=train_target_items,
    )
    test_dataset = Dataset(
        sequences=sequences,
        clv_dict=test_clv_dict,
        target_items=test_target_items,
    )
    return train_dataset, test_dataset, user_embedding_weight


def run_one_round(
    model: Model,
    dataloader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    args: Args,
    item_size: int,
    is_eval: bool = False,
) -> dict:
    result: dict[str, dict] = {
        "target": {"loss": 0.0, "y_pred": [], "y_true": []},
        "clv": {"loss": 0.0, "y_pred": [], "y_true": []},
    }

    if is_eval:
        model.eval()
    else:
        model.train()

    for batch in tqdm(dataloader):
        user_ids = []
        item_indices = []
        target_labels = []
        clv_labels = []

        for (user_id, user_context_items), target_items, clv in batch:
            context_items = []
            for _ in range(args.context_item_size):
                context_items.append(
                    user_context_items[random.randrange(len(user_context_items))]
                )

            # 正例
            for _ in range(min(args.sample_size, len(target_items))):
                v = list(context_items)
                v.append(target_items[random.randrange(0, len(target_items))])
                user_ids.append(user_id)
                item_indices.append(v)
                target_labels.append(1)
                clv_labels.append(clv)

            # 負例
            for _ in range(args.sample_size):
                v = list(context_items)
                v.append(random.randrange(item_size))
                user_ids.append(user_id)
                item_indices.append(v)
                target_labels.append(0)
                clv_labels.append(clv)

        user_ids = torch.LongTensor(user_ids).to(args.device)
        item_indices = torch.LongTensor(item_indices).to(args.device)
        target_labels = torch.FloatTensor(target_labels).to(args.device)
        clv_labels = torch.FloatTensor(clv_labels).to(args.device)
        y_clv, y_target = model.forward(user_ids, item_indices)

        target_loss = torch.nn.functional.binary_cross_entropy(
            y_target, target_labels
        ) / target_labels.size(0)
        clv_loss = torch.sqrt(torch.nn.functional.mse_loss(y_clv, clv_labels))
        loss = target_loss * args.alpha + clv_loss * (1 - args.alpha)

        if not is_eval:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        result["target"]["loss"] += target_loss.item()
        result["target"]["y_pred"].extend(y_target.tolist())
        result["target"]["y_true"].extend(target_labels.tolist())
        result["clv"]["loss"] += clv_loss.item()
        result["clv"]["y_pred"].extend(y_clv.tolist())
        result["clv"]["y_true"].extend(clv_labels.tolist())

    for term_name in result.keys():
        result[term_name]["loss"] /= len(dataloader)

    print(
        f"[loss] target_loss: {result['target']['loss']}, clv_loss: {result['clv']['loss']}"
    )
    print(
        f"[alpha_weighted_loss] target_loss: {result['target']['loss'] * args.alpha}, clv_loss: {result['clv']['loss'] * (1 - args.alpha)}"
    )
    return result


def train(
    model: Model,
    train_dataset: Dataset,
    test_dataset: Dataset,
    args: Args,
    item_size: int,
) -> tuple[list[dict], list[dict]]:
    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
    )
    test_dataloader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: x,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_results = []
    test_results = []

    for epoch in range(args.epochs):
        print(f"[epoch: {epoch + 1}/{args.epochs}]")
        result = run_one_round(
            model, train_dataloader, optimizer=optimizer, args=args, item_size=item_size
        )
        train_results.append(result)

        result = run_one_round(
            model,
            test_dataloader,
            optimizer=optimizer,
            args=args,
            item_size=item_size,
            is_eval=True,
        )
        test_results.append(result)

    return train_results, test_results
