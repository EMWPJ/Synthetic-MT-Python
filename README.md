# SyntheticMT - Magnetotelluric Time Series Synthesis

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.8+-green.svg)](https://www.python.org/)

**English** | [中文](#中文)

## Overview

SyntheticMT is a Python library for synthesizing magnetotelluric (MT) time series based on forward modeling. It provides a complete implementation of the method described in:

> Wang P, Chen X, Zhang Y (2023) [Synthesizing magnetotelluric time series based on forward modeling](https://www.frontiersin.org/articles/10.3389/feart.2023.1086749/full). *Frontiers in Earth Science* 11:1086749.

This project is a Python reimplementation of the original [Delphi-based SyntheticMTTimeSeries](https://github.com/EMWPJ/SyntheticMTTimeSeries), designed to be more Pythonic, user-friendly, and suitable for integration with modern Python scientific computing workflows.

## Installation

```bash
git clone https://github.com/EMWPJ/Synthetic-MT-Python.git
cd Synthetic-MT-Python
pip install numpy pyside6
```

## Quick Start

```python
from src import SyntheticTimeSeries, SyntheticSchema, SyntheticMethod, create_test_site
from datetime import datetime

# Create test site from forward modeling data
site = create_test_site()

# Configure TS3 acquisition system (2400Hz sampling)
schema = SyntheticSchema.from_ts('TS3')

# Create synthesizer with default method (RANDOM_SEG_PARTIAL)
synth = SyntheticTimeSeries(schema, SyntheticMethod.RANDOM_SEG_PARTIAL)

# Generate 1 minute of data
t1 = datetime(2023, 1, 1, 0, 0, 0)
t2 = datetime(2023, 1, 1, 0, 1, 0)

ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)

print(f"Generated {len(ex)} samples")
print(f"Ex range: [{ex.min():.4f}, {ex.max():.4f}] V/m")
print(f"Hx range: [{hx.min():.6f}, {hx.max():.6f}] A/m")
```

## Project Structure

```
Synthetic-MT-Python/
├── src/
│   ├── __init__.py          # Package exports
│   ├── synthetic_mt.py      # Core synthesis algorithm
│   ├── phoenix.py           # Phoenix TSn/TBL file I/O
│   ├── cli.py               # Command-line interface
│   └── gui.py               # PySide6 GUI
├── examples/
│   └── test_synthetic.py    # Test examples
├── output/                   # Generated output files
└── README.md
```

## Core Features

### 1. Time Series Synthesis

Six synthesis methods are implemented to simulate natural source polarization variability:

| Method | Description |
|--------|-------------|
| `FIX` | No segmentation - single continuous segment |
| `FIXED_AVG` | Fixed-length averaging segments |
| `FIXED_AVG_WINDOWED` | Fixed-length with Hanning window |
| `RANDOM_SEG` | Random-length segments |
| `RANDOM_SEG_WINDOWED` | Random-length with Hanning window |
| `RANDOM_SEG_PARTIAL` | Random-length with partial window (default) |

### 2. Acquisition System Configurations

| Config | Sample Rate | Frequency Range |
|--------|-------------|-----------------|
| TS2 | 2400 Hz | 10 - 12000 Hz |
| TS3 | 2400 Hz | 1 - 1000 Hz |
| TS4 | 150 Hz | 0.1 - 10 Hz |
| TS5 | 15 Hz | 1e-6 - 1 Hz |

### 3. File Format Support

- **ModEM Format**: Load forward modeling results
- **Phoenix TSn/TBL**: Read/write Phoenix MTU time series data
- **CLB/CLC**: Instrument calibration files
- **Output**: GMT, CSV, NumPy formats

### 4. Noise Injection

```python
from src import NoiseInjector, NoiseConfig, NoiseType

# Gaussian noise
config = NoiseConfig(noise_type=NoiseType.GAUSSIAN, amplitude=0.1)
injector = NoiseInjector(config, sample_rate=2400)
ex_noisy, ey_noisy, hx_noisy, hy_noisy, hz_noisy = injector.add_noise(ex, ey, hx, hy, hz)

# Impulsive noise
config = NoiseConfig(noise_type=NoiseType.IMPULSIVE, amplitude=1.0, probability=0.001)

# Powerline interference (50/60Hz)
config = NoiseConfig(noise_type=NoiseType.POWERLINE, amplitude=0.5, frequency=50.0)
```

## Usage

### Command Line

```bash
# Generate time series
python -m src.cli generate --begin-time "2023-01-01 00:00:00" --end-time "2023-01-01 00:01:00"

# List synthesis methods
python -m src.cli methods

# List TS configurations
python -m src.cli configs
```

### GUI

```bash
python -m src.gui
```

### Python Script

```bash
python examples/test_synthetic.py
```

## Algorithm

The synthesis algorithm is based on:

1. **Frequency-to-Time Conversion**: E(t) = A·cos(2πft + φ)
2. **Source Simulation**: Two orthogonal polarized sources with random linear combination
3. **Segment Stitching**: Windowed concatenation to simulate time-varying polarization

```
┌─────────────────────────────────────────────────────────────┐
│  Forward Modeling (ModEM)                                   │
│  Zxx, Zxy, Zyx, Zyy, Tzx, Tzy                            │
│  Ex1, Ey1, Hx1, Hy1, Hz1                                 │
│  Ex2, Ey2, Hx2, Hy2, Hz2                                   │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Natural Magnetic Field Scaling                            │
│  B_natural(f) = f^(-1/5) scaling                          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Segment Synthesis (6 methods)                             │
│  For each frequency:                                       │
│    - Generate E1(t), E2(t) from two polarizations         │
│    - Random amplitude/phase modulation                     │
│    - Window and concatenate segments                       │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Output                                                    │
│  Ex(t), Ey(t), Hx(t), Hy(t), Hz(t)                      │
└─────────────────────────────────────────────────────────────┘
```

## API Reference

### Core Classes

- `SyntheticTimeSeries`: Main synthesis engine
- `SyntheticSchema`: Configuration for TS acquisition systems
- `ForwardSite`: Forward modeling site data
- `EMFields`: Electromagnetic field data at one frequency

### I/O Functions

- `load_modem_file()`: Load ModEM forward results
- `save_gmt_timeseries()`: Save in GMT-compatible format
- `save_csv_timeseries()`: Save in CSV format
- `save_numpy_timeseries()`: Save in NumPy format

## Citation

If you use this software in your research, please cite:

```bibtex
@article{wang2023synthesizing,
  title={Synthesizing magnetotelluric time series based on forward modeling},
  author={Wang, Peijie and Chen, Xuan and Zhang, Yong},
  journal={Frontiers in Earth Science},
  volume={11},
  pages={1086749},
  year={2023},
  publisher={Frontiers}
}
```

## Related Projects

- [SyntheticMTTimeSeries](https://github.com/EMWPJ/SyntheticMTTimeSeries) - Original Delphi implementation
- [Modified-ModEM](https://github.com/EMWPJ/Modified-ModEM-with-output-electromagnetic-field-values) - ModEM with EM field output

## License

[GPL-3.0 License](LICENSE)

---

## 中文

# SyntheticMT - 大地电磁合成时间序列 Python版

基于论文: Wang P, Chen X, Zhang Y (2023) [基于正演模拟合成大地电磁时间序列](https://www.frontiersin.org/articles/10.3389/feart.2023.1086749/full). *Frontiers in Earth Science* 11:1086749.

## 安装

```bash
git clone https://github.com/EMWPJ/Synthetic-MT-Python.git
cd Synthetic-MT-Python
pip install numpy pyside6
```

## 快速开始

```python
from src import SyntheticTimeSeries, SyntheticSchema, SyntheticMethod, create_test_site
from datetime import datetime

site = create_test_site()
schema = SyntheticSchema.from_ts('TS3')
synth = SyntheticTimeSeries(schema, SyntheticMethod.RANDOM_SEG_PARTIAL)

t1 = datetime(2023, 1, 1, 0, 0, 0)
t2 = datetime(2023, 1, 1, 0, 1, 0)

ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)
```

## 运行方式

- **命令行**: `python -m src.cli generate --begin-time "2023-01-01 00:00:00" --end-time "2023-01-01 00:01:00"`
- **图形界面**: `python -m src.gui`
- **Python脚本**: `python examples/test_synthetic.py`

## 主要功能

- 6种合成方法
- ModEM正演结果解析
- Phoenix TSn/TBL文件读写
- 噪声注入（高斯、冲击、工频干扰）
- GMT/CSV/NumPy输出格式
- 自然磁场强度计算
- PySide6图形界面
