import numpy as np
import torch


def set_global_seeds(seed: int = 42) -> None:
    """Sets seed for Optuna objective to ensures reproducibility between trials."""

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False