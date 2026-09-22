import json
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import numpy.typing as npt
import numpy as np
from typing import Any, TextIO


@dataclass(slots=True)
class TrainingTracker:
    """
    Represents a tracker object for tracking training performance.
    Allows dynamic attribute generation.
    """

    config: dict[str, Any]
    loss_weights: dict[str, float]
    best_epoch: int = -1

    _metrics: dict[str, list[Any]] = field(
        default_factory=lambda: defaultdict(list)
    )

    def __getitem__(self, key: str) -> list[Any]:
        if hasattr(self, key) and not key.startswith("_"):
            val = getattr(self, key)
            if isinstance(val, list):
                return val
            
        return self._metrics[key]

    def extend(self, key: str, val: list[Any]) -> None:
        self[key].extend(val)

    def to_dict(self) -> dict[str, Any]:
        base_dict = {
            "config": self.config,
            "loss_weights": self.loss_weights,
            "best_epoch": self.best_epoch,
        }
        base_dict.update(self._metrics)

        return base_dict

    def to_json(self, file: TextIO) -> None:        
        json.dump(self.to_dict(), file, indent=2)


@dataclass(slots=True)
class ScoresStats:
    """
    Represents an object storing scoring statistics. 
    Allows dynamic attribute generation.
    """

    min: float
    max: float
    median: float
    mean: float
    std: float

    _stats: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float:
        if hasattr(self, key) and not key.startswith("_"):
            val = getattr(self, key)
            if isinstance(val, float):
                return val
            
        return self._stats[key]

    def __setitem__(self, key: str, value: float) -> None:
        if hasattr(self, key) and not key.startswith("_"):
            setattr(self, key, value)
        else:
            self._stats[key] = value

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(slots=True)
class TuneTemperatureResult:
    """
    Represents a result object for tuning inference temperature for a given 
    temperature.
    """

    scores_mean: float
    scores_std: float
    scores_median: float
    clean_scores_median: float
    prelim_threshold: float
    clean_samples: int

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)
    

@dataclass(slots=True)
class CalibrationResult:
    """
    Represents a result object for calibrating inference temperature and 
    threshold.
    Allows dynamic attribute generation.
    """

    country: str
    device: str
    cont_w: float
    cat_w: float
    method: str
    cal_window_days: int

    _metrics: dict[str, float | list[float] | dict[str, Any]] = field(default_factory=dict)

    def __getitem__(self, key: str) -> float | list[float] | dict[str, Any]:
        if hasattr(self, key) and not key.startswith("_"):
            val = getattr(self, key)
            if isinstance(val, (float, list, dict)):
                return val
            
        return self._metrics[key]

    def __setitem__(self, key: str, value: float | list[float] | dict[str, Any]) -> None:
        if hasattr(self, key) and not key.startswith("_"):
            setattr(self, key, value)
        else:
            self._metrics[key] = value

    def to_dict(self) -> dict[str, Any]:
        base_dict = {
            "country": self.country,
            "device": self.device,
            "cont_w": self.cont_w,
            "cat_w": self.cat_w,
            "method": self.method,
            "cal_window_days": self.cal_window_days
        }
        base_dict.update(self._metrics)

        return base_dict

    def to_json(self, file: TextIO) -> None:        
        json.dump(self.to_dict(), file, indent=2)


@dataclass(slots=True)
class MTPredictionResult:
    """
    Represents a result object for predictions of multi-task model.
    Allows dynamic attribute generation.
    """

    scores: npt.NDArray[np.float32]
    l3_pred: dict[str, npt.NDArray[np.float32]] | None = None
    l7_pred: dict[str, npt.NDArray[np.float32]] | None = None
    at_pred: npt.NDArray[np.float32] | None = None
    at_conf: npt.NDArray[np.float32] | None = None
    loss_total: npt.NDArray[np.float32] | None = None
    loss_l3: npt.NDArray[np.float32] | None = None
    loss_l7: npt.NDArray[np.float32] | None = None
    loss_at: npt.NDArray[np.float32] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class EvaluationResult:
    """
    Represents a result object for applying models to make predictions.
    Allows dynamic attribute generation.
    """

    scores: npt.NDArray[np.float32]
    threshold: float
    mask: npt.NDArray[np.bool_]
    anom_starts: npt.NDArray[np.int64]
    anom_ends: npt.NDArray[np.int64]

    _metrics: dict[str, npt.NDArray[np.float32]] = field(default_factory=dict)

    def __getitem__(self, key: str) -> npt.NDArray[np.float32]:
        if hasattr(self, key) and not key.startswith("_"):
            val = getattr(self, key)
            if isinstance(val, np.ndarray):
                return val
            
        return self._metrics[key]

    def __setitem__(self, key: str, value: npt.NDArray[np.float32]) -> None:
        if hasattr(self, key) and not key.startswith("_"):
            setattr(self, key, value)
        else:
            self._metrics[key] = value
