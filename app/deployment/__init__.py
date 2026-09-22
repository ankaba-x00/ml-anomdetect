# app/deployment/__init__.py
"""
deployment package
=================
this package provides core utilities for inference
- schema : fetch result object for inference
"""

from .schema import RegionTimeseriesFetchResult

__all__ = ["RegionTimeseriesFetchResult"]