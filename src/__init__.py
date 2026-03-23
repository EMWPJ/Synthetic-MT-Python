"""
SyntheticMT - 大地电磁合成时间序列库

基于论文: Wang P, Chen X, Zhang Y (2023) 
Synthesizing magnetotelluric time series based on forward modeling
Front. Earth Sci. 11:1086749
"""

from .synthetic_mt import (
    SegmentMethod,
    SEGMENT_METHOD_NAMES,
    EMFields,
    Site,
    TimeSeriesGenerator,
    MTSchema,
    nature_field_amplitude,
    single_freq_signal,
    hanning,
    inv_hanning,
)

from .phoenix import TsnFile, TblFile

__all__ = [
    'SegmentMethod',
    'SEGMENT_METHOD_NAMES', 
    'EMFields',
    'Site',
    'TimeSeriesGenerator',
    'MTSchema',
    'TsnFile',
    'TblFile',
    'nature_field_amplitude',
    'single_freq_signal',
    'hanning',
    'inv_hanning',
]
