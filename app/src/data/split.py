import numpy as np
from typing import Iterator


def timeseries_seq_split(
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    train_ratio: float = 0.75,
    val_ratio: float = 0.10,
) -> tuple[
    tuple[np.ndarray, np.ndarray],  # train
    tuple[np.ndarray, np.ndarray],  # val
    tuple[np.ndarray, np.ndarray],  # test
]:
    """
    Deterministic sequential split where timeseries data is split chronologically
        train set : first 70% of data
        val set : next 15% of data
        test set : final 15% of data
    """

    assert len(X_cont) == len(X_cat), "Continuous and categorical row-mismatch!"

    N = len(X_cont)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train = (X_cont[:n_train], X_cat[:n_train])
    val = (X_cont[n_train:n_train + n_val], X_cat[n_train:n_train + n_val])
    test = (X_cont[n_train + n_val:], X_cat[n_train + n_val:])

    return train, val, test

def timeseries_cv_splits(
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    n_splits: int = 4,
    min_train_ratio: float = 0.50,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
) -> Iterator[
    tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]
]:
    """
    Rolling timeseries cross-validation generator where fixed-size train set slides forward
        0 ---- train ----|-- val --|- test -
                shift →→→→→→→→→→→→→→→

    Ensures:
       - train set always chronologically before val and test
       - train never gets too small (min_train_ratio)

    Returns:
        (train_cont, train_cat), (val_cont, val_cat), (test_cont, test_cat)
    """

    assert len(X_cont) == len(X_cat), "Continuous and categorical row-mismatch!"

    N = len(X_cont)
    min_train = int(N * min_train_ratio)
    val_len   = int(N * val_ratio)
    test_len  = int(N * test_ratio)

    total_eval = val_len + test_len
    if min_train + total_eval >= N:
        raise ValueError("Not enough samples to support these ratios.")

    max_offset = N - (min_train + total_eval)
    offsets = np.linspace(0, max_offset, n_splits, dtype=int)

    for offset in offsets:
        train_end = offset + min_train
        val_end   = train_end + val_len
        test_end  = val_end + test_len

        train = (X_cont[offset:train_end],   X_cat[offset:train_end])
        val   = (X_cont[train_end:val_end], X_cat[train_end:val_end])
        test  = (X_cont[val_end:test_end],   X_cat[val_end:test_end])
        yield train, val, test
