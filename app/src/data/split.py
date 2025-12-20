import numpy as np
from typing import Iterator, Optional


def timeseries_seq_split(
    D1: np.ndarray,
    D2: Optional[np.ndarray] = None,
    train_ratio: float = 0.75,
    val_ratio: float = 0.10,
) -> tuple[
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
        tuple[np.ndarray, np.ndarray],
    ]:
    """
    Deterministic sequential split where timeseries data is split chronologically
        train set : first 70% of data
        val set : next 15% of data
        test set : final 15% of data
    """
    N = len(D1)
    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)
    
    if D2 is None:
        train = D1[:n_train]
        val = D1[n_train:n_train + n_val]
        test = D1[n_train + n_val:]
    else:
        assert len(D1) == len(D2), "[ERROR] Continuous and categorical row-mismatch!"

        train = (D1[:n_train], D2[:n_train])
        val = (D1[n_train:n_train + n_val], D2[n_train:n_train + n_val])
        test = (D1[n_train + n_val:], D2[n_train + n_val:])

    return train, val, test

def timeseries_cv_splits(
    D1: np.ndarray,
    D2: np.ndarray,
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

    assert len(D1) == len(D2), "Continuous and categorical row-mismatch!"

    N = len(D1)
    min_train = int(N * min_train_ratio)
    val_len = int(N * val_ratio)
    test_len = int(N * test_ratio)

    total_eval = val_len + test_len
    if min_train + total_eval >= N:
        raise ValueError("Not enough samples to support these ratios.")

    max_offset = N - (min_train + total_eval)
    offsets = np.linspace(0, max_offset, n_splits, dtype=int)

    for offset in offsets:
        train_end = offset + min_train
        val_end = train_end + val_len
        test_end = val_end + test_len

        train = (D1[offset:train_end], D2[offset:train_end])
        val = (D1[train_end:val_end], D2[train_end:val_end])
        test = (D1[val_end:test_end], D2[val_end:test_end])

        yield train, val, test
