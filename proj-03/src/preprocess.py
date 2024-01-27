import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder


def preprocess_dunnhumby() -> None:
    """
    table: transaction

    | user_id | t | store_id | item_id |
    |---------|---|----------|---------|
    | 2375    | 0 | 0        | 0       |
    | 2375    | 0 | 1        | 1       |
    | 2375    | 0 | 1        | 1       |
    | 2375    | 1 | 0        | 1       |
    | 2380    | 0 | 1        | 0       |

    table: recommendation

    | t | store_id | item_ids     |
    |---|----------|--------------|
    | 0 | 0        | [0, 1, 3, 5] |
    | 1 | 0        | [0, 3, 4, 5] |
    | 3 | 1        | [2, 4, 5, 6] |
    """

    def data_path(file: str) -> str:
        return f"../data/dunnhumby_The-Complete-Journey/dunnhumby_The-Complete-Journey CSV/{file}"

    transaction_df = pd.read_csv(data_path("transaction_data.csv"))
    causal_df = pd.read_csv(data_path("causal_data.csv"))

    # 推薦があった週だけ残す
    valid_weeks = set(causal_df.WEEK_NO.unique())
    transaction_df = transaction_df[transaction_df.WEEK_NO.isin(valid_weeks)]
    causal_df = causal_df[causal_df.WEEK_NO.isin(valid_weeks)]

    causal_df = causal_df[causal_df.display != "0"]
    # shops that have at least one visitor for each week
    a = transaction_df.groupby("STORE_ID")["WEEK_NO"].nunique()
    valid_stores = set(a[a == transaction_df["WEEK_NO"].nunique()].index)
    transaction_df = transaction_df[transaction_df["STORE_ID"].isin(valid_stores)]
    causal_df = causal_df[causal_df["STORE_ID"].isin(valid_stores)]

    # items recommended for at least one week on average among the shops
    a = causal_df["PRODUCT_ID"].value_counts()
    valid_items1 = set(a[a >= len(valid_stores)].index)

    # items that existed for at least half the period (47 weeks)
    a = transaction_df.groupby("PRODUCT_ID")["WEEK_NO"].nunique()
    valid_items2 = set(a[a > transaction_df["WEEK_NO"].nunique() / 2].index)

    # users visiting more than one store in at least five weeks
    a = transaction_df.groupby(["household_key", "WEEK_NO"])["STORE_ID"].nunique()
    a = a[a > 1].groupby("household_key").count()
    valid_users = set(a[a >= 5].index)

    valid_items = valid_items1 & valid_items2
    print(len(valid_users), len(valid_items), len(valid_stores))

    transaction_df = transaction_df[
        transaction_df.PRODUCT_ID.isin(valid_items)
        & transaction_df.household_key.isin(valid_users)
    ].reset_index(drop=True)
    causal_df = causal_df[causal_df.PRODUCT_ID.isin(valid_items)].reset_index(drop=True)

    transaction_df = transaction_df[
        ["household_key", "WEEK_NO", "STORE_ID", "PRODUCT_ID"]
    ]
    causal_df = (
        causal_df.groupby(["WEEK_NO", "STORE_ID"])["PRODUCT_ID"]
        .unique()
        .rename("PRODUCT_IDS")
        .to_frame()
        .reset_index()
    )

    causal_df["PRODUCT_IDS"] = causal_df["PRODUCT_IDS"].apply(
        lambda s: " ".join(map(str, s))
    )
    transaction_df = transaction_df.rename(
        columns={
            "household_key": "user_id",
            "WEEK_NO": "t",
            "STORE_ID": "store_id",
            "PRODUCT_ID": "item_id",
        }
    )
    causal_df = causal_df.rename(
        columns={"WEEK_NO": "t", "STORE_ID": "store_id", "PRODUCT_IDS": "item_ids"}
    )
    transaction_df.to_csv("../data/preprocessed/transaction.csv", index=False)
    causal_df.to_csv("../data/preprocessed/recommendation.csv", index=False)


def convert_to_dataset() -> None:
    transaction_df = pd.read_csv("../data/preprocessed/transaction.csv")
    recommendation_df = pd.read_csv(
        "../data/preprocessed/recommendation.csv", index_col=["t", "store_id"]
    )
    user_le = LabelEncoder().fit(transaction_df.user_id)
    item_le = LabelEncoder().fit(transaction_df.item_id)

    item_ids = set(item_le.classes_)

    transaction_df.user_id = user_le.fit_transform(transaction_df.user_id)
    transaction_df.item_id = item_le.fit_transform(transaction_df.item_id)

    recommendation_df.item_ids = (
        recommendation_df.item_ids.apply(lambda s: list(map(int, s.split())))
        .apply(lambda s: list(filter(lambda p: p in item_ids, s)))
        .apply(lambda s: set(item_le.transform(s)))
    )

    td = transaction_df.t.max()
    te = 8

    rec_dict = recommendation_df.item_ids.to_dict()

    data = {}

    for col_name, (tl, tr) in {
        "valid_train_purchased_items": (1, td - 2 * te),
        "valid_eval_purchased_items": (td - 2 * te + 1, td - te),
        "test_train_purchased_items": (te + 1, td - te),
        "test_eval_purchased_items": (td - te + 1, td),
    }.items():
        data[col_name] = (
            transaction_df[(tl <= transaction_df.t) & (transaction_df.t <= tr)]
            .groupby("user_id")["item_id"]
            .agg(set)
            .apply(lambda e: list(e))
            .rename(col_name)
        )

    recommend_dict = {
        "valid_train_recommended_items": (1, td - 2 * te),
        "valid_eval_recommended_items": (td - 2 * te + 1, td - te),
        "test_train_recommended_items": (te + 1, td - te),
        "test_eval_recommended_items": (td - te + 1, td),
    }

    for col_name, _ in recommend_dict.items():
        data[col_name] = []

    for _, user_df in transaction_df.drop_duplicates(
        subset=["user_id", "t", "store_id"]
    ).groupby("user_id")[["t", "store_id"]]:
        for col_name, (tl, tr) in recommend_dict.items():
            data[col_name].append(set())

        for _, row in user_df.iterrows():
            for col_name, (tl, tr) in recommend_dict.items():
                if tl <= row.t <= tr and (row.t, row.store_id) in rec_dict:
                    data[col_name][-1] |= rec_dict[row.t, row.store_id]

        for col_name, _ in recommend_dict.items():
            data[col_name][-1] = list(data[col_name][-1])

    df = pd.DataFrame(data)

    df.to_json("../data/preprocessed/preprocessed.json", orient="index")

    params = {"user_n": len(user_le.classes_), "item_n": len(item_le.classes_)}
    with open("../data/preprocessed/param.json", "w") as f:
        json.dump(params, f)


if __name__ == "__main__":
    preprocess_dunnhumby()
    convert_to_dataset()
