import json

import matplotlib.pyplot as plt
import pandas as pd
import torch

from src.lib.util import *

plt.style.use("ggplot")


def main() -> None:
    df = pd.read_json("./data/preprocessed/preprocessed.json", orient="index")
    with open("./data/preprocessed/param.json", "r") as f:
        params = json.load(f)
        print("params:", params)

    user_n, item_n = params["user_n"], params["item_n"]

    args = Args.from_args()  # type: ignore
    print("args:", args)
    model, losses, results = train_nn(df=df, user_n=user_n, item_n=item_n, args=args)

    torch.save(model, f"./model/{args.model}.pt")

    plt.plot(losses)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.show()

    add_record(args, results[-1])


if __name__ == "__main__":
    main()
