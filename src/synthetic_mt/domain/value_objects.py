"""Domain value objects - enums and immutable configurations."""
from enum import Enum
from dataclasses import dataclass


class SyntheticMethod(Enum):
    """合成方法"""
    FIX = 0
    FIXED_AVG = 1
    FIXED_AVG_WINDOWED = 2
    RANDOM_SEG = 3
    RANDOM_SEG_WINDOWED = 4
    RANDOM_SEG_PARTIAL = 5


SYNTHETIC_METHOD_NAMES = {
    SyntheticMethod.FIX: 'No Segment',
    SyntheticMethod.FIXED_AVG: 'Fixed Length Average',
    SyntheticMethod.FIXED_AVG_WINDOWED: 'Fixed Length Average & Windowed',
    SyntheticMethod.RANDOM_SEG: 'Random Segment Length',
    SyntheticMethod.RANDOM_SEG_WINDOWED: 'Random Segment Length & Windowed',
    SyntheticMethod.RANDOM_SEG_PARTIAL: 'Random Segment Length & Partially Windowed',
}


class NoiseType(Enum):
    """噪声类型"""
    SQUARE_WAVE = 'square'
    TRIANGULAR = 'triangular'
    IMPULSIVE = 'impulsive'
    GAUSSIAN = 'gaussian'
    POWERLINE = 'powerline'


@dataclass
class NoiseConfig:
    """噪声配置"""
    noise_type: NoiseType = NoiseType.GAUSSIAN
    amplitude: float = 0.0
    frequency: float = 0.0
    probability: float = 0.01
    phase: float = 0.0


TS_CONFIGS = {
    'TS2': {'sample_rate': 2400, 'freq_min': 10, 'freq_max': 12000},
    'TS3': {'sample_rate': 2400, 'freq_min': 1, 'freq_max': 1000},
    'TS4': {'sample_rate': 150, 'freq_min': 0.1, 'freq_max': 10},
    'TS5': {'sample_rate': 15, 'freq_min': 1e-6, 'freq_max': 1},
}
