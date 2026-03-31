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
│   ├── synthetic_mt/          # Core synthesis library (modular structure)
│   │   ├── domain/          # Domain entities & services
│   │   ├── infrastructure/  # I/O handlers
│   │   ├── application/     # Use cases
│   │   └── presentation/    # GUI
│   └── ...
├── examples/
│   ├── test_synthetic.py    # Basic examples
│   └── mt_workflow/         # 1D MT workflow (模块化架构)
│       ├── backend/         # 后台算法核心
│       │   ├── core.py      # 算法实现 (Model, Forward, Processor)
│       │   ├── api.py       # API接口
│       │   └── verify_all.py # 模块验证
│       ├── gui_simple.py    # 简化版GUI (单窗口, 调用backend.api)
│       ├── gui_workflow.py  # 多标签页GUI (1D正演/合成/处理)
│       ├── gui.py          # 旧版GUI
│       └── ...
├── docs/                    # Documentation
└── README.md
```

## 1D MT Workflow (模块化架构)

完整的1D大地电磁正演合成与处理工作流，采用**后台算法与界面分离**架构。

### 架构

```
GUI (gui_workflow.py)  →  API层 (backend/api.py)  →  算法核心 (backend/core.py)
     │                         │                           │
   多标签页界面                接口                       MT1DModel
   图表显示                  MTWorkflowAPI              MT1DForward
   事件处理                  单例模式                   TimeSeriesSynthesizer
                                                      TimeSeriesProcessor
                                                      Model1DValidator
```

### 两种GUI

| GUI | 文件 | 描述 |
|-----|------|------|
| **多标签页GUI** | `gui_workflow.py` | 推荐使用，三标签页界面（1D正演/合成/数据处理） |
| 简化版GUI | `gui_simple.py` | 单窗口界面，适合简单工作流 |

### 两种合成方法

#### 1. 确定性合成 (`synthesize_time_series_deterministic`)
- 保持精确E-H相位关系: `Ex = Zxy * Hy`
- 用于算法验证
- 处理结果与正演精确匹配 (<1%误差)

#### 2. 随机合成 (`synthesize_time_series_random`)
- 实现论文的 `RANDOM_SEG_PARTIAL` 算法
- 每分段随机振幅 (`RandG(1,1)`) 和相位 (`Random * 2π`)
- 模拟自然源变化
- 处理结果有较大误差（预期行为）

### 快速开始

```python
# 使用API进行完整工作流
from examples.mt_workflow.backend.api import get_api

api = get_api()

# 1. 创建模型
api.get_preset_model("uniform_100")

# 2. 正演计算
forward_result = api.run_forward()

# 3. 合成时间序列
ts_result = api.synthesize_time_series("TS3", duration=10)

# 4. 处理时间序列
processed = api.process_time_series()

# 5. 对比结果
comparator = api.compare_results()
```

### 后台验证

```bash
cd examples/mt_workflow
python backend/verify_all.py
```

### 启动GUI

```bash
cd examples/mt_workflow
python gui_simple.py
```

### 核心模块

| 模块 | 功能 |
|------|------|
| `backend/core.py` | 算法核心: MT1DModel, MT1DForward, TimeSeriesSynthesizer, TimeSeriesProcessor |
| `backend/api.py` | 统一API: MTWorkflowAPI单例, get_api()访问 |
| `backend/verify_all.py` | 后台验证脚本 |
| `gui_simple.py` | 简化版GUI (纯前端, 调用API) |

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

### 1D MT Workflow API

```python
from examples.mt_workflow.backend.api import get_api

api = get_api()
api.get_preset_model("uniform_100")
forward_result = api.run_forward()
ts_result = api.synthesize_time_series_deterministic("TS3", duration=60)
processed = api.process_time_series()
comparator = api.compare_results()
```

### I/O Functions

- `load_modem_file()`: Load ModEM forward results
- `save_gmt_timeseries()`: Save in GMT-compatible format
- `save_csv_timeseries()`: Save in CSV format
- `save_numpy_timeseries()`: Save in NumPy format

## Full Pipeline Test

Run the complete 1D MT workflow test:

```bash
cd examples/mt_workflow
python backend/test_full_pipeline.py
```

This tests:
1. **Forward Modeling**: 1D model → impedance
2. **Deterministic Synthesis**: Impedance → time series (preserves E-H phase)
3. **FFT Processing**: Time series → estimated impedance
4. **Comparison**: Forward vs Processed results

**Test Results** (Deterministic Synthesis):
| Model | Max rho_a Error | Mean rho_a Error | Max Phase Error | Status |
|-------|---------------|-----------------|-----------------|--------|
| uniform_100 | 0.93% | 0.30% | 0.39° | PASS |
| uniform_1000 | 0.93% | 0.30% | 0.39° | PASS |
| two_layer_hl | 1.14% | 0.35% | 0.48° | PASS |
| two_layer_ll | 0.32% | 0.12% | 0.11° | PASS |
| three_layer_hll | 0.60% | 0.24% | 0.21° | PASS |

**Note**: Random synthesis (RANDOM_SEG_PARTIAL) produces larger errors (2-100%) due to intentional random amplitude/phase perturbations, which is the expected behavior per the paper.

**Generated Artifacts** (in `docs/plots/`):
- `<model>_comparison.png` - Per-model comparison
- `all_models_summary.png` - Summary plot
- `error_summary.png` - Error analysis
- `test_report.md` - Test report

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

## 1D MT工作流 (模块化)

完整的1D大地电磁正演合成与处理工作流，采用后台与界面分离架构：

- **后台**: `examples/mt_workflow/backend/` - 算法核心独立验证
- **前端**: `examples/mt_workflow/gui_simple.py` - PySide6 GUI
- **验证**: `python backend/verify_all.py` - 所有模块正确性验证

### 预设模型

| 模型名 | 描述 |
|--------|------|
| `uniform_100` | 均匀半空间 100 Ω·m |
| `uniform_1000` | 均匀半空间 1000 Ω·m |
| `two_layer_hl` | 两层高阻-低阻 (1000→10 Ω·m) |
| `two_layer_ll` | 两层低阻-高阻 (10→1000 Ω·m) |
| `three_layer_hll` | 三层高-低-低 (1000→10→100 Ω·m) |
