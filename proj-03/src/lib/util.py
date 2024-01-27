import abc
import dataclasses
import datetime
import json
import pathlib
import random
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm

from lib.config import *
from lib.model import MatrixFactorization, Model
from lib.sampler import *


def eval(
    rec_list: list[list[int]],
    df: pd.DataFrame,
    args: Args,
) -> dict[str, float]:
    acc = 0
    uplift = 0.0
    target_user_n = 0

    result = {}

    for k in TOP_K:
        for u, dict_u in df.iterrows():
            if dict_u[f"{args.mode}_eval_purchased_items"] is None:
                continue
            L_M = set(rec_list[u][:k])
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


def train(
    df: pd.DataFrame,
    user_n: int,
    item_n: int,
    args: Args,
) -> tuple[torch.nn.Module, list[float], list[dict]]:
    set_seed(seed=args.seed)

    if args.model == "mf":
        model: torch.nn.Module = MatrixFactorization(
            d_model=args.d, user_n=user_n, item_n=item_n
        ).to(device=torch.device(args.device))
    elif args.model == "nn":
        model = Model(d_model=args.d, user_n=user_n, item_n=item_n).to(
            device=torch.device(args.device)
        )
    else:
        assert False

    if args.sampling == "uplift-based-pointwise":
        sampler: Sampler = UpliftBasedPointwiseSampler(
            user_n=user_n, item_n=item_n, df=df
        )
    elif args.sampling == "accuracy-based-pointwise":
        sampler = AccuracyBasedPointwiseSampler(user_n=user_n, item_n=item_n, df=df)
    else:
        assert False

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.eta, weight_decay=args.lmda
    )
    losses = []
    results = []

    for t in range(args.epochs):
        model.train()
        u_list, i_list, r_ui_list = sampler.sample(
            args=args,
        )
        optimizer.zero_grad()
        y = model.forward(
            u=torch.LongTensor(u_list).to(device=args.device),
            i=torch.LongTensor(i_list).to(device=args.device),
        )
        r_ui = torch.FloatTensor(r_ui_list).to(device=args.device)
        loss = criterion(y, r_ui)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        print(
            f"[{t+1:{len(str(args.epochs))}}/{args.epochs}] {loss.item():.5}",
            end="\r",
        )

        if (t + 1) % args.eval_step == 0:
            result = eval_model(
                model=model, df=df, user_n=user_n, item_n=item_n, args=args
            )
            print(result)
            results.append(result)

    return model, losses, results


def eval_model(
    model: torch.nn.Module, df: pd.DataFrame, user_n: int, item_n: int, args: Args
) -> dict:
    model.eval()

    rec_list = []
    for u in tqdm.tqdm(range(user_n)):
        y = model.forward(
            u=torch.LongTensor([u] * item_n).to(device=args.device),
            i=torch.arange(item_n).to(device=args.device),
        )
        rec_list.append(y.argsort().cpu().detach().numpy()[::-1][: max(TOP_K)].tolist())

    return eval(rec_list=rec_list, df=df, args=args)


def add_record(args: Args, evaluations: dict, log_dir: str = "log/") -> None:
    record = {"args": vars(args), "evaluations": evaluations}
    log_path = pathlib.Path(log_dir)
    if args.exp_name is not None:
        log_path /= args.exp_name
    log_path.mkdir(parents=True, exist_ok=True)
    log_path /= f"{datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')}.json"
    with open(log_path, "w") as f:
        json.dump(record, f, indent=4, sort_keys=True)


def visualize_losses(
    losses: list[float],
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    fig, ax = plt.subplots()
    ax.plot(losses)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    return fig, ax


def visualize_results(
    results: list[dict],
    args: Args,
) -> tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    df = pd.DataFrame(results)
    df["epoch"] = (df.index + 1) * args.eval_step
    fig, ax = plt.subplots()
    df.plot(x="epoch", ax=ax)
    ax.set_xlabel("epoch")
    return fig, ax


def set_seed(seed: int) -> None:
    # random
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.mps.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
