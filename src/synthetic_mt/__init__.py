"""
SyntheticMT - Backward Compatibility Package

This package provides backward compatibility for code that imports from `synthetic_mt`.
All public APIs are re-exports from the new DDD-structured modules.

New code should import directly from the domain/infrastructure/application modules:
    from src.domain.entities import EMFields, ForwardSite
    from src.domain.services.synthesis import SyntheticTimeSeries
    etc.
"""

from .compat import *

__all__ = compat.__all__
