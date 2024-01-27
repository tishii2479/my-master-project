import classopt

TOP_K = [10, 30, 100]


@classopt.classopt(default_long=True)
class Args:
    exp_name: str = None  # type: ignore
    mode: str = "valid"
    model: str = "mf"
    sampling: str = "uplift-based-pointwise"
    seed: int = 0
    alpha: float = 0.6
    gamma_p: float = 0.2
    gamma_r: float = 0.5
    eta: float = 1e-2
    lmda: float = 1e-8
    d: int = 100
    batch_size: int = 1_000
    eval_step: int = 1_000
    epochs: int = 5_000
    device: str = "cpu"
