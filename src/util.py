import collections
import random
from dataclasses import dataclass
from typing import Optional

import matplotlib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from src.model import *


@dataclass
class Args:
    d_model: int = 32
    dim_feedforward: int = 64
    batch_size: int = 64
    nhead: int = 4
    num_layers: int = 4
    epochs: int = 10
    lr: float = 1e-3
    sample_size: int = 5
    negative_sample_size: int = 5
    alpha: float = 0.5
    context_item_size: int = 10
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
        self.user_idx = []
        self.sequences = []
        self.target_items = []
        self.clv = []

        for user_idx, sequence in tqdm(sequences.items()):
            self.user_idx.append(user_idx)
            self.sequences.append(sequence)
            self.target_items.append(
                target_items[user_idx] if user_idx in target_items else []
            )
            self.clv.append(clv_dict[user_idx] if user_idx in clv_dict else 0)

    def __len__(self) -> int:
        return len(self.user_idx)

    def __getitem__(self, idx: int) -> tuple[int, list[int], list[int], float]:
        return (
            self.user_idx[idx],
            self.sequences[idx],
            self.target_items[idx],
            self.clv[idx],
        )


class NegativeSampler:
    def __init__(
        self, sequences: dict[int, list[int]], item_size: int, power: float = 0.75
    ) -> None:
        self.item_size = item_size

        counts: collections.Counter = collections.Counter()
        for sequence in sequences.values():
            for item in sequence:
                counts[item] += 1

        self.word_p = np.zeros(self.item_size)
        for i in range(self.item_size):
            self.word_p[i] = max(1, counts[i])

        self.word_p = np.power(self.word_p, power)
        self.word_p /= np.sum(self.word_p)

    def sample(self, shape: int | tuple) -> np.ndarray:
        # 正解ラベルが含まれていても無視する
        negative_sample = np.random.choice(
            self.item_size,
            size=shape,
            replace=True,
            p=self.word_p,
        )
        return negative_sample


def load_interaction_df(
    last_review_date: pd.Timestamp | str,
    train_split_date: pd.Timestamp | str,
    padding_token: str = "[pad]",
    mask_token: Optional[str] = None,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    interaction_df = pd.read_csv(
        "../data/ml-25m/ratings.csv", dtype={"userId": str, "movieId": str}
    ).rename(
        columns={"userId": "user_id", "movieId": "item_id"},
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
        interaction_df=interaction_df,
        padding_token=padding_token,
        mask_token=mask_token,
    )
    return interaction_df, user_le, item_le


def create_user_features(
    feature_df: pd.DataFrame, split_date: pd.Timestamp | str
) -> pd.DataFrame:
    # TODO: 上側95%点で切る
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


def create_targets(target_df: pd.DataFrame) -> tuple[dict, dict, dict]:
    # CLV
    clv = target_df.user_id.value_counts().sort_index()
    threshold = clv.quantile(0.95)
    clv /= threshold
    clv[clv >= 1] = 1
    clv_dict = clv.to_dict()

    # 離反
    churn = clv.copy(deep=True)
    churn[churn >= 0] = 1
    churn_dict = churn.to_dict()

    # CV商品
    target_items = target_df.groupby("user_id").item_id.agg(list).to_dict()

    return clv_dict, churn_dict, target_items


def encode_user_item_id(
    interaction_df: pd.DataFrame,
    padding_token: str = "[pad]",
    mask_token: Optional[str] = None,
) -> tuple[pd.DataFrame, LabelEncoder, LabelEncoder]:
    user_le = LabelEncoder().fit(interaction_df.user_id)
    items = interaction_df.item_id.values.tolist() + [padding_token]
    if mask_token is not None:
        items.append(mask_token)
    item_le = LabelEncoder().fit(items)

    interaction_df.user_id = user_le.transform(interaction_df.user_id)
    interaction_df.item_id = item_le.transform(interaction_df.item_id)

    return interaction_df, user_le, item_le


def create_dataset(
    interaction_df: pd.DataFrame,
    train_split_date: pd.Timestamp | str,
    test_split_date: pd.Timestamp | str,
    item_size: int,
) -> tuple[Dataset, Dataset, np.ndarray, np.ndarray, NegativeSampler]:
    # TODO: refactor
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

    train_sequences = feature_df.groupby("user_id").item_id.agg(list).to_dict()
    test_sequences = (
        pd.concat([feature_df, target_df])
        .groupby("user_id")
        .item_id.agg(list)
        .to_dict()
    )
    train_user_feature_table = create_user_features(
        feature_df=feature_df, split_date=train_split_date
    ).values.astype(np.float32)
    test_user_feature_table = create_user_features(
        feature_df=pd.concat([feature_df, target_df]).reset_index(drop=True),
        split_date=test_split_date,
    ).values.astype(np.float32)

    _, train_churn_dict, train_target_items = create_targets(target_df=target_df)
    _, test_churn_dict, test_target_items = create_targets(target_df=test_df)

    train_dataset = Dataset(
        sequences=train_sequences,
        clv_dict=train_churn_dict,
        target_items=train_target_items,
    )
    test_dataset = Dataset(
        sequences=test_sequences,
        clv_dict=test_churn_dict,
        target_items=test_target_items,
    )

    negative_sampler = NegativeSampler(sequences=test_sequences, item_size=item_size)

    return (
        train_dataset,
        test_dataset,
        train_user_feature_table,
        test_user_feature_table,
        negative_sampler,
    )


def plot_loss(
    train_results: list[dict],
    test_results: list[dict],
    loss_name: str,
    ax: matplotlib.axes.Axes,
) -> tuple[list, list]:
    train_losses = list(map(lambda r: r[loss_name]["loss"], train_results))
    test_losses = list(map(lambda r: r[loss_name]["loss"], test_results))
    ax.plot(train_losses, label="train")
    ax.plot(test_losses, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{loss_name}_loss")
    ax.legend()
    ax.grid()

    return train_losses, test_losses


def plot_r2_score(
    train_results: list[dict],
    test_results: list[dict],
    loss_name: str,
    ax: matplotlib.axes.Axes,
) -> tuple[list, list]:
    train_r2_scores = list(
        map(
            lambda r: r2_score(r[loss_name]["y_true"], r[loss_name]["y_pred"]),
            train_results,
        )
    )
    test_r2_scores = list(
        map(
            lambda r: r2_score(r[loss_name]["y_true"], r[loss_name]["y_pred"]),
            test_results,
        )
    )

    ax.plot(train_r2_scores, label="train")
    ax.plot(test_r2_scores, label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{loss_name}_r2_score")
    ax.legend()
    ax.grid()

    return train_r2_scores, test_r2_scores


def plot_roc_auc(
    train_result: dict, test_result: dict, loss_name: str, ax: matplotlib.axes.Axes
) -> tuple[list, list]:
    fpr, tpr, _ = roc_curve(
        train_result[loss_name]["y_true"], train_result[loss_name]["y_pred"]
    )
    train_auc = roc_auc_score(
        train_result[loss_name]["y_true"], train_result[loss_name]["y_pred"]
    )
    ax.plot(fpr, tpr, label=f"train = {train_auc:.5f}")

    fpr, tpr, _ = roc_curve(
        test_result[loss_name]["y_true"], test_result[loss_name]["y_pred"]
    )
    test_auc = roc_auc_score(
        test_result[loss_name]["y_true"], test_result[loss_name]["y_pred"]
    )
    ax.plot(fpr, tpr, label=f"test  = {test_auc:.5f}")

    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend()
    ax.grid()

    return train_auc, test_auc


def plot_auc(
    train_results: list[dict],
    test_results: list[dict],
    loss_name: str,
    ax: matplotlib.axes.Axes,
) -> tuple[list, list]:
    train_aucs = list(
        map(
            lambda r: roc_auc_score(r[loss_name]["y_true"], r[loss_name]["y_pred"]),
            train_results,
        )
    )
    test_aucs = list(
        map(
            lambda r: roc_auc_score(r[loss_name]["y_true"], r[loss_name]["y_pred"]),
            test_results,
        )
    )

    ax.plot(train_aucs, label="train")
    ax.plot(test_aucs, label="test")

    ax.set_xlabel("epoch")
    ax.set_ylabel(f"{loss_name}_auc")
    ax.legend()
    ax.grid()

    return train_aucs, test_aucs


def calc_hit_ratio(target_items: list[int], recommendation: list[int], k: int) -> float:
    if len(set(target_items) & set(recommendation[:k])) > 0:
        return 1
    return 0


def calc_precision(target_items: list[int], recommendation: list[int], k: int) -> float:
    return len(set(target_items) & set(recommendation[:k])) / k


def calc_recall(target_items: list[int], recommendation: list[int], k: int) -> float:
    return len(set(target_items) & set(recommendation[:k])) / len(target_items)


def calc_mrr(target_items: list[int], recommendation: list[int], k: int) -> float:
    s = set(target_items)
    for i, item in enumerate(recommendation[:k]):
        if item in s:
            return 1 / (i + 1)
    return 0


def calc_map(target_items: list[int], recommendation: list[int], k: int) -> float:
    s = set(target_items)
    precision_sum = 0.0
    for i, item in enumerate(recommendation[:k]):
        if item in s:
            precision_sum += calc_precision(target_items, recommendation, i + 1)
    return precision_sum / len(target_items)
