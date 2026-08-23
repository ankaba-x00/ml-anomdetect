import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def unsupervised_dataloader(
    X_f: np.ndarray, 
    X_i: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Build DataLoader with
       X_f : FloatType (float32) features
       X_i : IntegerType (int64) features 
    """
    Xc_f = torch.from_numpy(X_f.astype(np.float32))
    Xk_i = torch.from_numpy(X_i.astype(np.int64))

    Tds = TensorDataset(Xc_f, Xk_i)

    return DataLoader(
        Tds, 
        batch_size=batch_size, 
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available()
    )

def supervised_dataloader(
    X_cont: np.ndarray,
    X_cat: np.ndarray,
    y_l3: np.ndarray,
    y_l7: np.ndarray,
    y_attack: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    Xc = torch.from_numpy(X_cont.astype(np.float32))
    Xk = torch.from_numpy(X_cat.astype(np.int64))
    y3 = torch.from_numpy(y_l3.astype(np.float32))
    y7 = torch.from_numpy(y_l7.astype(np.float32))
    ya = torch.from_numpy(y_attack.astype(np.int64))

    ds = TensorDataset(Xc, Xk, y3, y7, ya)

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available(),
    )