# app/src/data/processing/__init__.py
"""
data processing package
=======================
this package provides core utilities for data processing stage
- params : dataset key parameters
- split : methods for timeseries data splitting
"""

from .params import DSFILE_MAP, FIELD_MAP
from .split import timeseries_seq_split, timeseries_cv_splits

__all__ = ["DSFILE_MAP", "FIELD_MAP", "timeseries_seq_split", "timeseries_cv_splits"]