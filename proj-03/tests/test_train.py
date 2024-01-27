import json

import matplotlib.pyplot as plt
import pandas as pd

from lib.config import Args
from lib.util import eval_model, train

plt.style.use("ggplot")


def test_train() -> None:
    df = pd.read_json("./data/preprocessed/preprocessed.json", orient="index")
    with open("./data/preprocessed/param.json", "r") as f:
        params = json.load(f)
        print("params:", params)

    user_n, item_n = params["user_n"], params["item_n"]

    args = Args(epochs=100, eval_step=20)
    print("args:", args)

    model, losses, results = train(df=df, user_n=user_n, item_n=item_n, args=args)
    eval_model(model=model, df=df, user_n=user_n, item_n=item_n, args=args)
