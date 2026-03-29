"""
后台模块

backend/
├── core.py    - 算法核心实现
├── api.py     - API接口层
└── verify_all.py - 验证脚本
"""

from .core import (
    MT1DModel,
    MT1DForward,
    TimeSeriesSynthesizer,
    DeterministicTimeSeriesSynthesizer,
    TimeSeriesProcessor,
    Model1DValidator,
    ResultsComparator,
    TSConfig,
    TS_CONFIGS,
    get_default_forward_periods,
    get_default_processing_periods,
    MU0,
)

from .api import MTWorkflowAPI, get_api, reset_api

__all__ = [
    # Core
    "MT1DModel",
    "MT1DForward",
    "TimeSeriesSynthesizer",
    "DeterministicTimeSeriesSynthesizer",
    "TimeSeriesProcessor",
    "Model1DValidator",
    "ResultsComparator",
    "TSConfig",
    "TS_CONFIGS",
    "get_default_forward_periods",
    "get_default_processing_periods",
    "MU0",
    # API
    "MTWorkflowAPI",
    "get_api",
    "reset_api",
]
