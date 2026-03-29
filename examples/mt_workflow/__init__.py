"""
MT 1D 正演合成与处理工作流

模块化设计:
- config: 采集系统配置
- model_1d: 1D地电模型定义
- forward_1d: 1D正演计算
- synthesizer: 时间序列合成
- processor: 时间序列处理 (FFT)
- validator: 结果验证与对比

用法:
    from mt_workflow import run_simple_workflow
    results = run_simple_workflow()
"""

from .config import TSConfig, TS_CONFIGS, get_config, MULTI_BAND_CONFIG, MU0
from .model_1d import (
    MT1DModel,
    create_uniform_halfspace,
    get_preset_model,
    PRESET_MODELS,
)
from .forward_1d import MT1DForward, compute_theoretical_response
from .synthesizer import TimeSeriesSynthesizer, MultiBandSynthesizer
from .processor import TimeSeriesProcessor, SpectrumResult
from .validator import Model1DValidator, ResultsComparator, ValidationResult
from .main import run_simple_workflow, run_single_band_workflow, run_multi_band_workflow

__all__ = [
    # config
    "TSConfig",
    "TS_CONFIGS",
    "get_config",
    "MULTI_BAND_CONFIG",
    "MU0",
    # model
    "MT1DModel",
    "create_uniform_halfspace",
    "get_preset_model",
    "PRESET_MODELS",
    # forward
    "MT1DForward",
    "compute_theoretical_response",
    # synthesizer
    "TimeSeriesSynthesizer",
    "MultiBandSynthesizer",
    # processor
    "TimeSeriesProcessor",
    "SpectrumResult",
    # validator
    "Model1DValidator",
    "ResultsComparator",
    "ValidationResult",
    # main
    "run_simple_workflow",
    "run_single_band_workflow",
    "run_multi_band_workflow",
]
