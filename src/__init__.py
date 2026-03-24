"""
SyntheticMT - 大地电磁合成时间序列库

基于论文: Wang P, Chen X, Zhang Y (2023) 
Synthesizing magnetotelluric time series based on forward modeling
Front. Earth Sci. 11:1086749

This module provides backward-compatible imports. For new code, import directly from:
    from synthetic_mt.domain.entities import EMFields, ForwardSite
    from synthetic_mt.domain.value_objects import SyntheticMethod, NoiseType
    from synthetic_mt.domain.services.synthesis import SyntheticTimeSeries
    from synthetic_mt.infrastructure.io.phoenix import TsnFile, TblFile
"""

# Domain layer
from .domain.entities import EMFields, ForwardSite, nature_magnetic_amplitude
from .domain.value_objects import (
    SyntheticMethod,
    SYNTHETIC_METHOD_NAMES,
    NoiseType,
    NoiseConfig,
    TS_CONFIGS,
)
from .domain.services.synthesis import (
    freq_to_time,
    hanning_window,
    inv_hanning_window,
    SynthesisSchema,
    SyntheticTimeSeries,
    calculate_mt_scale_factors,
    create_test_site,
    load_modem_file,
)
from .domain.services.noise import NoiseInjector, add_powerline_interference
from .domain.services.calibration import (
    CalibrationData,
    SystemCalibrator,
    ClbFile,
    ClcFile,
)

# Infrastructure layer
from .infrastructure.io.modem import ModemReader
from .infrastructure.io.phoenix import TsnFile, TblFile, TagInfo
from .infrastructure.io.output import (
    save_gmt_timeseries,
    save_csv_timeseries,
    save_numpy_timeseries,
    load_numpy_timeseries,
)

# Application layer
from .application import SynthesisUseCase, SynthesisRequest, SynthesisResult

# Presentation layer
try:
    from .presentation.gui import SyntheticMTGui
except ImportError:
    SyntheticMTGui = None

__all__ = [
    # Entities
    'EMFields',
    'ForwardSite',
    'nature_magnetic_amplitude',
    # Value objects
    'SyntheticMethod',
    'SYNTHETIC_METHOD_NAMES',
    'NoiseType',
    'NoiseConfig',
    'TS_CONFIGS',
    # Synthesis
    'freq_to_time',
    'hanning_window',
    'inv_hanning_window',
    'SynthesisSchema',
    'SyntheticTimeSeries',
    'calculate_mt_scale_factors',
    'create_test_site',
    'load_modem_file',
    # Noise
    'NoiseInjector',
    'add_powerline_interference',
    # Calibration
    'CalibrationData',
    'SystemCalibrator',
    'ClbFile',
    'ClcFile',
    # Infrastructure
    'ModemReader',
    'TsnFile',
    'TblFile',
    'TagInfo',
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
