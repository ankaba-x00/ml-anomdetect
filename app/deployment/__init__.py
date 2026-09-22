# app/deployment/__init__.py
"""
inference module
=================
this module provides core utilities for using a packaged model bundle for inference
- schema : fetch result object for inference
"""

from .schema import RegionTimeseriesFetchResult

__all__ = ["RegionTimeseriesFetchResult"]