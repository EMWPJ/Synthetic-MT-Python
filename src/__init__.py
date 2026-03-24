"""
SyntheticMT - 大地电磁合成时间序列库

基于论文: Wang P, Chen X, Zhang Y (2023) 
Synthesizing magnetotelluric time series based on forward modeling
Front. Earth Sci. 11:1086749
"""

from .domain import (
    SyntheticMethod,
    SYNTHETIC_METHOD_NAMES,
    NoiseType,
    NoiseConfig,
    TS_CONFIGS,
)

from .synthetic_mt import (
    EMFields,
    ForwardSite,
    SyntheticSchema,
    SyntheticTimeSeries,
    load_modem_file,
    create_test_site,
    NoiseInjector,
    add_powerline_interference,
    CalibrationData,
    ClbFile,
    ClcFile,
    nature_magnetic_amplitude,
    calculate_mt_scale_factors,
    save_gmt_timeseries,
    save_csv_timeseries,
    save_numpy_timeseries,
    load_numpy_timeseries,
    SystemCalibrator,
)

from .phoenix import TsnFile, TblFile

from .gui import SyntheticMTGui

__all__ = [
    'SyntheticMethod',
    'SYNTHETIC_METHOD_NAMES',
    'EMFields',
    'ForwardSite',
    'SyntheticSchema',
    'SyntheticTimeSeries',
    'TS_CONFIGS',
    'load_modem_file',
    'create_test_site',
    'NoiseType',
    'NoiseConfig',
    'NoiseInjector',
    'add_powerline_interference',
    'CalibrationData',
    'ClbFile',
    'ClcFile',
    'nature_magnetic_amplitude',
    'calculate_mt_scale_factors',
    'save_gmt_timeseries',
    'save_csv_timeseries',
    'save_numpy_timeseries',
    'load_numpy_timeseries',
    'SystemCalibrator',
    'TsnFile',
    'TblFile',
    'SyntheticMTGui',
]
