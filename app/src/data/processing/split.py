import numpy as np
import numpy.typing as npt
from typing import Iterator


def timeseries_seq_split(
    arrays: list[npt.NDArray],
    train_ratio: float = 0.75,
    val_ratio: float = 0.10,
) -> tuple[list[npt.NDArray], list[npt.NDArray], list[npt.NDArray]]:
    """
    Splits timeseries chronologically depending on ratios provided in order of train, val and test set.
        0 ---- train ----|-- val --|- test -
    """
    if not arrays:
        raise ValueError("[ERROR] Timeseries split requires at least one array")

    N = len(arrays[0])
    if any(len(arr) != N for arr in arrays):
        raise ValueError("[ERROR] Timeseries split requires arrays of equal length")

    n_train = int(N * train_ratio)
    n_val = int(N * val_ratio)

    train = [arr[:n_train] for arr in arrays]
    val = [arr[n_train : n_train + n_val] for arr in arrays]
    test = [arr[n_train + n_val :] for arr in arrays]

    return train, val, test

def timeseries_cv_splits(
    arrays: list[npt.NDArray],
    n_splits: int = 4,
    min_train_ratio: float = 0.50,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
) -> Iterator[list[list[npt.NDArray]]]:
    """
    Split timeseries by rolling fixed-size train set slides forward. Cross-validation ensures that train set is always chronologically before val and test.
        0 ---- train ----|-- val --|- test -
                shift →→→→→→→→→→→→→→→
    """

    if not arrays:
        raise ValueError("[ERROR] Timeseries split requires at least one array")

    N = len(arrays[0])
    if any(len(arr) != N for arr in arrays):
        raise ValueError("[ERROR] Timeseries split requires arrays of equal length")
    
    min_train = int(N * min_train_ratio)
    val_len = int(N * val_ratio)
    test_len = int(N * test_ratio)

    total_eval = val_len + test_len
    if min_train + total_eval >= N:
        raise ValueError("[ERROR] Not enough datapoints to satisfy timeseries split ratios")

    max_offset = N - (min_train + total_eval)
    offsets = np.linspace(0, max_offset, n_splits, dtype=int)

    for offset in offsets:
        train_end = offset + min_train
        val_end = train_end + val_len
        test_end = val_end + test_len

        train = [arr[offset:train_end] for arr in arrays]
        val = [arr[train_end:val_end] for arr in arrays]
        test = [arr[val_end:test_end] for arr in arrays]

        yield [train, val, test]
