"""Domain package - DDD value objects and entities."""
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
