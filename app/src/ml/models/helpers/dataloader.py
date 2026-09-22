import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import DataLoader, TensorDataset


def unsupervised_dataloader(
    X_f: npt.NDArray[np.float32],
    X_i: npt.NDArray[np.int64],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Builds DataLoader with
       Xc : FloatType (float32) features
       Xk : IntegerType (int64) features 
    """
    Xc = torch.from_numpy(X_f.astype(np.float32))
    Xk = torch.from_numpy(X_i.astype(np.int64))

    Tds = TensorDataset(Xc, Xk)

    return DataLoader(
        Tds, 
        batch_size=batch_size, 
        shuffle=shuffle,
        pin_memory=torch.cuda.is_available()
    )

def supervised_dataloader(
    X_cont: npt.NDArray[np.float32],
    X_cat: npt.NDArray[np.int64],
    y_l3: npt.NDArray[np.float32],
    y_l7: npt.NDArray[np.float32],
    y_attack: npt.NDArray[np.int64],
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    """
    Builds DataLoader with
        Xc : FloatType (float32) features
        Xk : IntegerType (int64) features 
        y7 : FloatType (float32) label
        y3 : FloatType (float32) label
        ya : IntegerType (int64) label 
    """
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