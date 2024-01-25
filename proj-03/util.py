import abc
import dataclasses
import datetime
import json
import pathlib
from typing import Optional

import numpy as np
import pandas as pd


@dataclasses.dataclass
class Args:
    exp_name: Optional[str] = None
    mode: str = "valid"
    sampling: str = "uplift-based-pointwise"
    seed: int = 0
    alpha: float = 0.6
    gamma_p: float = 0.2
    gamma_r: float = 0.5
    eta: float = 1e-2
    lmda: float = 1e-2
    d: int = 100
    batch_size: int = 1_000
    epochs: int = 10_000


class Sampler(abc.ABC):
    def __init__(self, user_n: int, item_n: int, df: pd.DataFrame) -> None:
        self.user_n = user_n
        self.item_n = item_n
        self.df_dict = df.to_dict(orient="index")

        df_set = df.copy()
        for col in df.columns:
            df_set[col] = df_set[col].apply(lambda x: set() if x is None else set(x))

        df_set_dict = df_set.to_dict(orient="index")
        self.df_set_dict = df_set_dict

    @abc.abstractmethod
    def sample(
        self, rnd: np.random.RandomState, args: Args
    ) -> tuple[list[int], list[int], list[int]]:
        raise NotImplementedError()


class AccuracyBasedPointwiseSampler(Sampler):
    def sample(
        self,
        rnd: np.random.RandomState,
        args: Args,
    ) -> tuple[list[int], list[int], list[int]]:
        u_list, i_list, r_ui_list = (
            [0] * args.batch_size,
            [0] * args.batch_size,
            [0] * args.batch_size,
        )

        for b in range(args.batch_size):
            C_P = 0
            C_NP = 1
            u = rnd.randint(0, self.user_n)
            C = rnd.choice([C_P, C_NP], p=[args.gamma_p, 1 - args.gamma_p])

            if C == C_P:
                r_ui = 1
                i = rnd.choice(self.df_dict[u][f"{args.mode}_train_purchased_items"])
            else:
                r_ui = 0
                while True:
                    i = rnd.randint(0, self.item_n)
                    if (
                        i
                        not in self.df_set_dict[u][f"{args.mode}_train_purchased_items"]
                    ):
                        break

            u_list[b], i_list[b], r_ui_list[b] = u, i, r_ui

        return u_list, i_list, r_ui_list


class UpliftBasedPointwiseSampler(Sampler):
    def sample(
        self,
        rnd: np.random.RandomState,
        args: Args,
    ) -> tuple[list[int], list[int], list[int]]:
        C_RP = 0
        C_NR_NP = 1
        C_other = 2
        p_C_RP = args.gamma_p * args.gamma_r
        p_C_NR_NP = (1 - args.gamma_p) * (1 - args.gamma_r)
        p_C_other = 1 - p_C_RP - p_C_NR_NP

        u_list, i_list, r_ui_list = (
            [0] * args.batch_size,
            [0] * args.batch_size,
            [0] * args.batch_size,
        )

        for b in range(args.batch_size):
            u = rnd.randint(0, self.user_n)
            C = rnd.choice([C_RP, C_NR_NP, C_other], p=[p_C_RP, p_C_NR_NP, p_C_other])

            if C == C_RP:
                r_ui = 1

                while True:
                    i = rnd.choice(
                        self.df_dict[u][f"{args.mode}_train_purchased_items"]
                    )
                    if i in self.df_set_dict[u][f"{args.mode}_train_recommended_items"]:
                        break
            elif C == C_NR_NP:
                if rnd.random() <= args.alpha:
                    r_ui = 1
                else:
                    r_ui = 0

                while True:
                    i = rnd.randint(0, self.item_n)
                    if (
                        i
                        not in self.df_set_dict[u][f"{args.mode}_train_purchased_items"]
                        and i
                        not in self.df_set_dict[u][
                            f"{args.mode}_train_recommended_items"
                        ]
                    ):
                        break
            else:
                r_ui = 0

                while True:
                    i = rnd.randint(0, self.item_n)
                    if (
                        i in self.df_set_dict[u][f"{args.mode}_train_purchased_items"]
                        and i
                        not in self.df_set_dict[u][
                            f"{args.mode}_train_recommended_items"
                        ]
                    ) or (
                        i
                        not in self.df_set_dict[u][f"{args.mode}_train_purchased_items"]
                        and i
                        in self.df_set_dict[u][f"{args.mode}_train_recommended_items"]
                    ):
                        break

            u_list[b], i_list[b], r_ui_list[b] = u, i, r_ui

        return u_list, i_list, r_ui_list


def eval(
    rec_list: list[list[int]],
    df: pd.DataFrame,
    top_k: list[int],
    args: Args,
) -> dict[str, float]:
    acc = 0
    uplift = 0.0
    target_user_n = 0

    result = {}

    for k in top_k:
        for u, dict_u in df.iterrows():
            if dict_u[f"{args.mode}_eval_purchased_items"] is None:
                continue
            L_M = set(rec_list[u])
            L_D = set(dict_u[f"{args.mode}_eval_recommended_items"])
            L_M_and_D = list(L_M & L_D)
            L_M_not_D = list(L_M - L_D)

            Y = set(dict_u[f"{args.mode}_eval_purchased_items"])

            if len(L_M_and_D) > 0 and len(L_M_not_D) > 0:
                tau = sum([1 if e in Y else 0 for e in L_M_and_D]) / len(
                    L_M_and_D
                ) - sum([1 if e in Y else 0 for e in L_M_not_D]) / len(L_M_not_D)
                uplift += tau

                v = len(L_M & Y)
                acc += v

                target_user_n += 1

        accuracy = acc / target_user_n / k
        uplift = uplift / target_user_n

        result[f"Accuracy@{k}"] = accuracy
        result[f"Uplift@{k}"] = uplift

    return result


def train_rmf(
    df: pd.DataFrame,
    user_n: int,
    item_n: int,
    args: Args,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    rnd = np.random.RandomState(args.seed)
    X_u = rnd.normal(size=(user_n, args.d))
    X_v = rnd.normal(size=(item_n, args.d))

    if args.sampling == "uplift-based-pointwise":
        sampler: Sampler = UpliftBasedPointwiseSampler(
            user_n=user_n, item_n=item_n, df=df
        )
    elif args.sampling == "accuracy-based-pointwise":
        sampler = AccuracyBasedPointwiseSampler(user_n=user_n, item_n=item_n, df=df)
    else:
        assert False

    losses = []

    for t in range(args.epochs):
        u, i, r_ui = sampler.sample(
            rnd=rnd,
            args=args,
        )

        x_ui = sigmoid((X_u[u] * X_v[i]).sum(axis=-1))
        r_ui = np.array(r_ui)

        eps = 1e-8
        L = -(
            r_ui * np.log(np.maximum(eps, x_ui))
            + (1 - r_ui) * np.log(np.maximum(eps, 1 - x_ui))
        ).mean()
        losses.append(L)
        print(
            f"[{t+1:{len(str(args.epochs))}}/{args.epochs}] {L:.5}",
            end="\r",
        )

        X_u[u] -= args.eta * (
            (x_ui - r_ui).reshape(-1, 1) * X_v[i] + 2 * args.lmda * X_u[u]
        )
        X_v[i] -= args.eta * (
            (x_ui - r_ui).reshape(-1, 1) * X_u[u] + 2 * args.lmda * X_v[i]
        )

    return X_u, X_v, losses


def add_record(args: Args, evaluations: dict) -> None:
    record = {"args": vars(args), "evaluations": evaluations}
    log_path = pathlib.Path("./log/")
    if args.exp_name is not None:
        log_path /= args.exp_name
    log_path.mkdir(parents=True, exist_ok=True)
    log_path /= f"{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json"
    with open(log_path, "w") as f:
        json.dump(record, f, indent=4, sort_keys=True)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))
