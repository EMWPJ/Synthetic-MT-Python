"""
Backward Compatibility Layer for SyntheticMT

This module re-exports all public APIs from their new DDD locations
to maintain backward compatibility with existing code that imports from `synthetic_mt`.

New code should import directly from the domain/infrastructure/application modules.
"""

# Domain Entities
from ..domain.entities import (
    EMFields,
    ForwardSite,
    nature_magnetic_amplitude,
)

# Domain Value Objects
from ..domain.value_objects import (
    SyntheticMethod,
    SYNTHETIC_METHOD_NAMES,
    NoiseType,
    NoiseConfig,
    TS_CONFIGS,
)

# Domain Services - Synthesis
from ..domain.services.synthesis import (
    freq_to_time,
    hanning_window,
    inv_hanning_window,
    SyntheticSchema,
    SyntheticTimeSeries,
    calculate_mt_scale_factors,
    create_test_site,
    load_modem_file,
)

# Domain Services - Noise
from ..domain.services.noise import (
    NoiseInjector,
    add_powerline_interference,
)

# Note: Calibration classes (CalibrationData, SystemCalibrator, ClbFile, ClcFile)
# are still in the legacy synthetic_mt.py module as domain.services.calibration
# hasn't been created yet. They will be migrated in a future refactoring.
from ..synthetic_mt import (
    CalibrationData,
    SystemCalibrator,
    ClbFile,
    ClcFile,
)

# Infrastructure - ModEM I/O
from ..infrastructure.io.modem import (
    ModEMReader,
)

# Infrastructure - Phoenix I/O
from ..infrastructure.io.phoenix import (
    TsnFile,
    TblFile,
)

# Infrastructure - Output Formats
from ..infrastructure.io.output import (
    save_gmt_timeseries,
    save_csv_timeseries,
    save_numpy_timeseries,
    load_numpy_timeseries,
)

# Application Layer
from ..application import (
    SynthesisUseCase,
    SynthesisRequest,
    SynthesisResult,
)

# Presentation Layer
try:
    from ..presentation import SyntheticMTGui
except ImportError:
    # GUI is optional (requires PySide6)
    SyntheticMTGui = None

__all__ = [
    # Domain Entities
    'EMFields',
    'ForwardSite',
    'nature_magnetic_amplitude',
    # Domain Value Objects
    'SyntheticMethod',
    'SYNTHETIC_METHOD_NAMES',
    'NoiseType',
    'NoiseConfig',
    'TS_CONFIGS',
    # Domain Services - Synthesis
    'freq_to_time',
    'hanning_window',
    'inv_hanning_window',
    'SyntheticSchema',
    'SyntheticTimeSeries',
    'calculate_mt_scale_factors',
    'create_test_site',
    'load_modem_file',
    # Domain Services - Noise
    'NoiseInjector',
    'add_powerline_interference',
    # Calibration (legacy)
    'CalibrationData',
    'SystemCalibrator',
    'ClbFile',
    'ClcFile',
    # Infrastructure - ModEM
    'ModemReader',
    # Infrastructure - Phoenix
    'TsnFile',
    'TblFile',
    # Infrastructure - Output
    'save_gmt_timeseries',
    'save_csv_timeseries',
    'save_numpy_timeseries',
    'load_numpy_timeseries',
    # Application
    'SynthesisUseCase',
    'SynthesisRequest',
    'SynthesisResult',
    # Presentation
    'SyntheticMTGui',
]
