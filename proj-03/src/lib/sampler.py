import abc

import numpy as np
import pandas as pd

from lib.config import Args


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
    def sample(self, args: Args) -> tuple[list[int], list[int], list[int]]:
        raise NotImplementedError()


class AccuracyBasedPointwiseSampler(Sampler):
    def sample(
        self,
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
            u = np.random.randint(0, self.user_n)
            C = np.random.choice([C_P, C_NP], p=[args.gamma_p, 1 - args.gamma_p])

            if C == C_P:
                r_ui = 1
                i = np.random.choice(
                    self.df_dict[u][f"{args.mode}_train_purchased_items"]
                )
            else:
                r_ui = 0
                while True:
                    i = np.random.randint(0, self.item_n)
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
            u = np.random.randint(0, self.user_n)
            C = np.random.choice(
                [C_RP, C_NR_NP, C_other], p=[p_C_RP, p_C_NR_NP, p_C_other]
            )

            if C == C_RP:
                r_ui = 1

                while True:
                    i = np.random.choice(
                        self.df_dict[u][f"{args.mode}_train_purchased_items"]
                    )
                    if i in self.df_set_dict[u][f"{args.mode}_train_recommended_items"]:
                        break
            elif C == C_NR_NP:
                if np.random.random() <= args.alpha:
                    r_ui = 1
                else:
                    r_ui = 0

                while True:
                    i = np.random.randint(0, self.item_n)
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
                    i = np.random.randint(0, self.item_n)
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
