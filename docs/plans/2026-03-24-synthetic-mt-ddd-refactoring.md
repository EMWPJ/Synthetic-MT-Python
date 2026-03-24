# SyntheticMT DDD Refactoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Refactor the monolithic `synthetic_mt.py` into a DDD-based modular architecture with clear separation of concerns.

**Architecture:** Domain-Driven Design with 4 layers: domain (entities, services), application (use cases, DTOs), infrastructure (IO, calibration), presentation (GUI, CLI). Each layer has clear responsibilities and dependencies only flow inward.

**Tech Stack:** Python 3.8+, NumPy, pytest, PySide6 (optional GUI), scipy (optional)

---

## Task 1: Create Directory Structure

**Files:**
- Create: `src/synthetic_mt/domain/__init__.py`
- Create: `src/synthetic_mt/domain/services/__init__.py`
- Create: `src/synthetic_mt/application/__init__.py`
- Create: `src/synthetic_mt/infrastructure/__init__.py`
- Create: `src/synthetic_mt/infrastructure/io/__init__.py`
- Create: `src/synthetic_mt/infrastructure/io/modem/__init__.py`
- Create: `src/synthetic_mt/infrastructure/io/phoenix/__init__.py`
- Create: `src/synthetic_mt/infrastructure/io/output/__init__.py`
- Create: `src/synthetic_mt/infrastructure/calibration/__init__.py`
- Create: `src/synthetic_mt/presentation/__init__.py`
- Create: `src/synthetic_mt/compat.py`

**Step 1: Create all directories**

```bash
mkdir -p src/synthetic_mt/domain/services
mkdir -p src/synthetic_mt/application
mkdir -p src/synthetic_mt/infrastructure/io/modem
mkdir -p src/synthetic_mt/infrastructure/io/phoenix
mkdir -p src/synthetic_mt/infrastructure/io/output
mkdir -p src/synthetic_mt/infrastructure/calibration
mkdir -p src/synthetic_mt/presentation
```

**Step 2: Create __init__.py files**

Each `__init__.py` should export the public API of that module.

**Step 3: Commit**

```bash
git add -A && git commit -m "chore: create DDD directory structure"
```

---

## Task 2: Extract Value Objects

**Files:**
- Create: `src/synthetic_mt/domain/value_objects.py`

**Step 1: Create value_objects.py with content from synthetic_mt.py**

Extract from `synthetic_mt.py` lines 1-60:
```python
"""Domain value objects - enums and immutable configurations."""

from enum import Enum

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
```

**Step 2: Add dataclass import**

```python
from dataclasses import dataclass
```

**Step 3: Commit**

```bash
git add src/synthetic_mt/domain/value_objects.py
git commit -m "feat(domain): extract value objects (SyntheticMethod, NoiseType, NoiseConfig)"
```

---

## Task 3: Extract EMFields and ForwardSite Entities

**Files:**
- Create: `src/synthetic_mt/domain/entities.py`

**Step 1: Create entities.py with EMFields and ForwardSite**

Extract from `synthetic_mt.py` lines 63-270:
```python
"""Domain entities - core business objects."""

from dataclasses import dataclass, field
from typing import List
import numpy as np

@dataclass
class EMFields:
    """单频率点的电磁场数据"""
    freq: float
    ex1: complex = complex(0, 0)
    ey1: complex = complex(0, 0)
    hx1: complex = complex(0, 0)
    hy1: complex = complex(0, 0)
    hz1: complex = complex(0, 0)
    ex2: complex = complex(0, 0)
    ey2: complex = complex(0, 0)
    hx2: complex = complex(0, 0)
    hy2: complex = complex(0, 0)
    hz2: complex = complex(0, 0)
    zxx: complex = complex(0, 0)
    zxy: complex = complex(0, 0)
    zyx: complex = complex(0, 0)
    zyy: complex = complex(0, 0)
    tzx: complex = complex(0, 0)
    tzy: complex = complex(0, 0)

@dataclass
class ForwardSite:
    """正演测点"""
    name: str
    x: float = 0.0
    y: float = 0.0
    fields: List[EMFields] = field(default_factory=list)
    
    def frequencies(self) -> np.ndarray:
        return np.array([f.freq for f in self.fields])
    
    def add_fields(self, f: EMFields):
        self.fields.append(f)
    
    # ... (all ForwardSite methods)
```

**Step 2: Include all ForwardSite methods from synthetic_mt.py**

Copy these methods from `synthetic_mt.py` lines 99-268:
- `update_nature_magnetic_amplitude()`
- `interpolation()`
- `negative_harmonic_factor()`
- `get_feh1()`
- `add_calibration()`

**Step 3: Commit**

```bash
git add src/synthetic_mt/domain/entities.py
git commit -m "feat(domain): extract entities (EMFields, ForwardSite)"
```

---

## Task 4: Extract Synthesis Service

**Files:**
- Create: `src/synthetic_mt/domain/services/synthesis.py`

**Step 1: Create synthesis.py with synthesis functions**

Extract from `synthetic_mt.py` lines 270-800 (functions and SynthesisSchema):

```python
"""Domain service - time series synthesis algorithms."""

import numpy as np
from typing import Optional
from .value_objects import SyntheticMethod, NoiseConfig

# Helper functions
def nature_magnetic_amplitude(freq: float) -> float:
    """Calculate natural magnetic field amplitude B(f)."""
    # ... (from synthetic_mt.py)

def freq_to_time(amp: float, phase: float, freq: float, 
                 sample_rate: float, n: int, 
                 output: Optional[np.ndarray] = None) -> np.ndarray:
    """Convert frequency domain to time domain."""
    # ... (from synthetic_mt.py)

def hanning_window(n: int) -> np.ndarray:
    """Generate Hanning window."""
    # ... (from synthetic_mt.py)

def inv_hanning_window(n: int) -> np.ndarray:
    """Inverse Hanning window."""
    # ... (from synthetic_mt.py)

class SynthesisSchema:
    """合成配置"""
    # ... (from synthetic_mt.py)

class SyntheticTimeSeries:
    """时间序列合成器"""
    # ... (from synthetic_mt.py)
```

**Step 2: Verify no circular imports**

The service should import only from `domain/value_objects.py`.

**Step 3: Commit**

```bash
git add src/synthetic_mt/domain/services/synthesis.py
git commit -m "feat(domain): extract synthesis service"
```

---

## Task 5: Extract Noise Service

**Files:**
- Create: `src/synthetic_mt/domain/services/noise.py`

**Step 1: Create noise.py with noise injection functions**

Extract from `synthetic_mt.py` (NoiseInjector class and noise functions):

```python
"""Domain service - noise injection algorithms."""

import numpy as np
from typing import Optional
from .value_objects import NoiseType, NoiseConfig

class NoiseInjector:
    """噪声注入器"""
    
    def __init__(self, config: NoiseConfig, seed: Optional[int] = None):
        # ... from synthetic_mt.py
    
    def inject(self, data: np.ndarray, sample_rate: float) -> np.ndarray:
        # ... from synthetic_mt.py
    
    def _inject_gaussian(self, data: np.ndarray) -> np.ndarray:
        # ...
    
    def _inject_square(self, data: np.ndarray, sample_rate: float) -> np.ndarray:
        # ...
    
    # ... other methods

def add_powerline_interference(data: np.ndarray, sample_rate: float,
                               freq: float = 50.0, 
                               amplitude: float = 1e-10) -> np.ndarray:
    """Add powerline interference (50/60Hz)."""
    # ... from synthetic_mt.py
```

**Step 2: Commit**

```bash
git add src/synthetic_mt/domain/services/noise.py
git commit -m "feat(domain): extract noise service"
```

---

## Task 6: Extract Calibration Service

**Files:**
- Create: `src/synthetic_mt/domain/services/calibration.py`

**Step 1: Create calibration.py**

Extract from `synthetic_mt.py` (calibration-related code):

```python
"""Domain service - calibration handling."""

import numpy as np
from dataclasses import dataclass
from typing import Optional

@dataclass
class CalibrationData:
    """标定数据"""
    # ... from synthetic_mt.py

class SystemCalibrator:
    """系统标定器"""
    
    def __init__(self, calibration_data: Optional[CalibrationData] = None):
        # ...
    
    def calibrate(self, site: 'ForwardSite') -> None:
        # ...
    
    def get_response(self, channel: int, freq: float) -> complex:
        # ...
```

**Step 2: Commit**

```bash
git add src/synthetic_mt/domain/services/calibration.py
git commit -m "feat(domain): extract calibration service"
```

---

## Task 7: Extract Application Layer

**Files:**
- Create: `src/synthetic_mt/application/dto.py`
- Create: `src/synthetic_mt/application/synthesis_use_case.py`

**Step 1: Create dto.py with request/response DTOs**

```python
"""Application layer - data transfer objects."""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

class OutputFormat(Enum):
    GMT = 'gmt'
    CSV = 'csv'
    NUMPY = 'numpy'

@dataclass
class SynthesisRequest:
    """合成请求"""
    modem_path: str
    ts_config: str = 'TS3'
    method: 'SyntheticMethod' = None  # Will be imported
    noise_config: 'NoiseConfig' = None  # Will be imported
    output_format: OutputFormat = OutputFormat.GMT
    seed: Optional[int] = None

@dataclass
class SynthesisResult:
    """合成结果"""
    ex: np.ndarray
    ey: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    hz: np.ndarray
    sample_rate: float
    duration: float
    metadata: dict
```

**Step 2: Create synthesis_use_case.py**

```python
"""Application layer - synthesis use case."""

from .dto import SynthesisRequest, SynthesisResult
from ..domain.entities import ForwardSite
from ..domain.services.synthesis import SynthesisService, load_modem_file
from ..infrastructure.io.output.gmt import GmtExporter

class SynthesisUseCase:
    """合成时间序列用例"""
    
    def execute(self, request: SynthesisRequest) -> SynthesisResult:
        # Orchestrate the synthesis process
        sites = load_modem_file(request.modem_path)
        # ... perform synthesis
        # ... export result
        return result
```

**Step 3: Commit**

```bash
git add src/synthetic_mt/application/
git commit -m "feat(application): add application layer (DTOs, use case)"
```

---

## Task 8: Extract ModEM Infrastructure

**Files:**
- Create: `src/synthetic_mt/infrastructure/io/modem/reader.py`
- Create: `src/synthetic_mt/infrastructure/io/modem/types.py`

**Step 1: Create types.py**

```python
"""ModEM file format types."""

from enum import Enum

class BlockType(Enum):
    FULL_IMPEDANCE = "Full_Impedance"
    TIPPER = "Tipper"
    EM_FIELDS = "EM_Fields"
```

**Step 2: Create reader.py**

Extract from `synthetic_mt.py` (ModEM loading code):

```python
"""ModEM file reader."""

import numpy as np
from ...domain.entities import ForwardSite

class ModemReader:
    """ModEM 文件读取"""
    
    def read(self, path: str) -> list[ForwardSite]:
        """Read ModEM format file and return list of ForwardSite."""
        # ... from synthetic_mt.py load_modem_file function
```

**Step 3: Commit**

```bash
git add src/synthetic_mt/infrastructure/io/modem/
git commit -m "feat(infrastructure): add ModEM reader"
```

---

## Task 9: Extract Phoenix Infrastructure

**Files:**
- Create: `src/synthetic_mt/infrastructure/io/phoenix/tsn.py`
- Create: `src/synthetic_mt/infrastructure/io/phoenix/tbl.py`
- Create: `src/synthetic_mt/infrastructure/io/phoenix/types.py`

**Step 1: Move phoenix.py content**

Move content from `src/phoenix.py` to the new phoenix module:
- `tsn.py` - TsnFile class
- `tbl.py` - TblFile class
- `types.py` - TagInfo dataclass

**Step 2: Commit**

```bash
git add src/synthetic_mt/infrastructure/io/phoenix/
git commit -m "feat(infrastructure): add Phoenix IO module"
```

---

## Task 10: Extract Output Infrastructure

**Files:**
- Create: `src/synthetic_mt/infrastructure/io/output/gmt.py`
- Create: `src/synthetic_mt/infrastructure/io/output/csv.py`
- Create: `src/synthetic_mt/infrastructure/io/output/numpy_io.py`

**Step 1: Create output modules**

Extract from `synthetic_mt.py` (save functions):
```python
"""GMT format exporter."""

import numpy as np
from ...domain.entities import ForwardSite

class GmtExporter:
    """GMT 格式导出"""
    
    def export(self, path: str, ex: np.ndarray, ey: np.ndarray,
               hx: np.ndarray, hy: np.ndarray, hz: np.ndarray,
               sample_rate: float) -> None:
        # ... from synthetic_mt.py
```

**Step 2: Commit**

```bash
git add src/synthetic_mt/infrastructure/io/output/
git commit -m "feat(infrastructure): add output exporters (GMT, CSV, NumPy)"
```

---

## Task 11: Extract Calibration Infrastructure

**Files:**
- Create: `src/synthetic_mt/infrastructure/calibration/clb.py`
- Create: `src/synthetic_mt/infrastructure/calibration/clc.py`

**Step 1: Move CLB/CLC classes**

Extract from `synthetic_mt.py` (ClbFile, ClcFile classes) to:
- `clb.py` - CLB file handling
- `clc.py` - CLC file handling

**Step 2: Commit**

```bash
git add src/synthetic_mt/infrastructure/calibration/
git commit -m "feat(infrastructure): add CLB/CLC calibration modules"
```

---

## Task 12: Move Presentation Layer

**Files:**
- Create: `src/synthetic_mt/presentation/gui.py`
- Create: `src/synthetic_mt/presentation/cli.py`

**Step 1: Copy gui.py and cli.py**

Copy from `src/gui.py` to `src/synthetic_mt/presentation/gui.py`  
Copy from `src/cli.py` to `src/synthetic_mt/presentation/cli.py`

**Step 2: Update imports**

Update imports in both files to use new module paths.

**Step 3: Commit**

```bash
git add src/synthetic_mt/presentation/
git commit -m "feat(presentation): add GUI and CLI presentation layer"
```

---

## Task 13: Create Compatibility Layer

**Files:**
- Create: `src/synthetic_mt/compat.py`
- Modify: `src/synthetic_mt/__init__.py`

**Step 1: Create compat.py**

```python
"""Backward compatibility layer."""

from .domain.entities import ForwardSite, EMFields
from .domain.value_objects import (
    SyntheticMethod, NoiseType, NoiseConfig, TS_CONFIGS,
    SYNTHETIC_METHOD_NAMES
)
from .domain.services.synthesis import (
    SynthesisSchema, SyntheticTimeSeries,
    nature_magnetic_amplitude, calculate_mt_scale_factors,
    load_modem_file, create_test_site
)
from .domain.services.noise import NoiseInjector, add_powerline_interference
from .domain.services.calibration import CalibrationData, SystemCalibrator
from .infrastructure.io.phoenix.tsn import TsnFile
from .infrastructure.io.phoenix.tbl import TblFile
from .infrastructure.io.output.gmt import save_gmt_timeseries
from .infrastructure.io.output.csv import save_csv_timeseries
from .infrastructure.io.output.numpy_io import save_numpy_timeseries, load_numpy_timeseries

# Backward compatible convenience functions
def load_modem_file_compat(path: str):
    """Compatibility wrapper for load_modem_file."""
    from .infrastructure.io.modem.reader import ModemReader
    reader = ModemReader()
    return reader.read(path)

def create_test_site_compat():
    """Compatibility wrapper for create_test_site."""
    # ... existing implementation
```

**Step 2: Update __init__.py**

```python
"""SyntheticMT - 大地电磁合成时间序列库 (DDD架构)"""

# Re-export everything from compat for backward compatibility
from .compat import *

__all__ = compat.__all__
```

**Step 3: Commit**

```bash
git add src/synthetic_mt/compat.py src/synthetic_mt/__init__.py
git commit -m "feat(compat): add backward compatibility layer"
```

---

## Task 14: Delete Old Files

**Files:**
- Delete: `src/synthetic_mt.py`
- Delete: `src/phoenix.py`

**Step 1: Remove old files**

```bash
rm src/synthetic_mt.py src/phoenix.py
```

**Step 2: Commit**

```bash
git add -A && git rm src/synthetic_mt.py src/phoenix.py
git commit -m "refactor!: remove legacy monolithic files"
```

---

## Task 15: Verification

**Files:**
- Test: `tests/test_synthetic_mt.py`

**Step 1: Run all tests**

```bash
pytest tests/ -v
```

Expected: All 51 tests pass

**Step 2: Verify imports work**

```bash
python -c "from synthetic_mt import ForwardSite, SyntheticMethod, NoiseConfig; print('OK')"
```

**Step 3: Verify backward compatibility**

```bash
python -c "
from synthetic_mt import load_modem_file, create_test_site
site = create_test_site()
print(f'Created site: {site.name}')
"
```

**Step 4: Commit final verification**

```bash
git commit -m "test: verify all 51 tests pass after refactoring"
```

---

## Summary

| Task | Description | Files |
|------|-------------|-------|
| 1 | Create directory structure | 12 new `__init__.py` files |
| 2 | Extract value objects | `domain/value_objects.py` |
| 3 | Extract entities | `domain/entities.py` |
| 4 | Extract synthesis service | `domain/services/synthesis.py` |
| 5 | Extract noise service | `domain/services/noise.py` |
| 6 | Extract calibration service | `domain/services/calibration.py` |
| 7 | Create application layer | `application/dto.py`, `application/synthesis_use_case.py` |
| 8 | Extract ModEM infrastructure | `infrastructure/io/modem/` |
| 9 | Extract Phoenix infrastructure | `infrastructure/io/phoenix/` |
| 10 | Extract output infrastructure | `infrastructure/io/output/` |
| 11 | Extract calibration infrastructure | `infrastructure/calibration/` |
| 12 | Move presentation layer | `presentation/gui.py`, `presentation/cli.py` |
| 13 | Create compatibility layer | `compat.py`, `__init__.py` |
| 14 | Delete old files | Remove legacy files |
| 15 | Verification | Run tests |

**Total: 15 tasks, ~30 commits**
