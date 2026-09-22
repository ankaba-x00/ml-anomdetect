# app/src/ml/models/configs/__init__.py
"""
ml models configs package
=========================
this package holds all model config dataclasses
"""

from .cae import AEConfig
from .cmtae import MTAEConfig
from .cvae import VAEConfig

__all__ = ["AEConfig", "MTAEConfig", "VAEConfig"]