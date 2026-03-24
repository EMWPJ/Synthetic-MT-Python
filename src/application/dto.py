"""Application layer - Data Transfer Objects.

DTOs are simple data containers that transfer data between layers.
They should not contain any business logic - only data and validation.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, TYPE_CHECKING
from enum import Enum

# Import numpy for array type hints at runtime
import numpy as np

if TYPE_CHECKING:
    # These imports are only for type checking, avoiding circular imports
    from ..domain.value_objects import SyntheticMethod, NoiseConfig


class OutputFormat(Enum):
    """Output format for synthesized time series"""
    GMT = 'gmt'      # GMT-compatible text format
    CSV = 'csv'      # CSV format
    NUMPY = 'numpy'  # NumPy binary format


@dataclass
class SynthesisRequest:
    """合成请求 - Request to synthesize MT time series.
    
    Attributes:
        modem_path: Path to ModEM forward modeling result file
        ts_config: Time series configuration name (TS2, TS3, TS4, TS5)
        method: Synthesis method to use (uses default if not specified)
        noise_config: Optional noise configuration to add noise to output
        output_format: Output format (GMT, CSV, NUMPY)
        seed: Optional random seed for reproducibility
    """
    modem_path: str
    ts_config: str = 'TS3'
    method: Optional['SyntheticMethod'] = None
    noise_config: Optional['NoiseConfig'] = None
    output_format: OutputFormat = OutputFormat.GMT
    seed: Optional[int] = None


@dataclass
class SynthesisResult:
    """合成结果 - Result of MT time series synthesis.
    
    Attributes:
        ex: Electric field X component (V/m)
        ey: Electric field Y component (V/m)
        hx: Magnetic field X component (A/m)
        hy: Magnetic field Y component (A/m)
        hz: Magnetic field Z component (A/m)
        sample_rate: Sampling rate in Hz
        duration: Duration of time series in seconds
        metadata: Additional metadata about the synthesis
    """
    ex: np.ndarray
    ey: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    hz: np.ndarray
    sample_rate: float
    duration: float
    metadata: Dict[str, Any] = field(default_factory=dict)
