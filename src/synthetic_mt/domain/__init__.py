"""Domain layer - Core business logic with no external dependencies.

This module contains the core domain entities, value objects, and business rules
that define the application's domain model. Domain layer should have no imports
from infrastructure, application, or presentation layers.
"""

from .value_objects import (
    SyntheticMethod,
    SYNTHETIC_METHOD_NAMES,
    NoiseType,
    NoiseConfig,
    TS_CONFIGS,
)
from .entities import (
    EMFields,
    ForwardSite,
    nature_magnetic_amplitude,
)

__all__ = [
    'SyntheticMethod',
    'SYNTHETIC_METHOD_NAMES',
    'NoiseType',
    'NoiseConfig',
    'TS_CONFIGS',
    'EMFields',
    'ForwardSite',
    'nature_magnetic_amplitude',
]
