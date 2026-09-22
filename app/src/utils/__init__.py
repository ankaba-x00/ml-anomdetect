# app/src/utils/__init__.py
"""
utils package
=================
this package provides development utilities
- decorators : optional decorator to time execution time
"""

from .decorators import timeit

__all__ = ["timeit"]