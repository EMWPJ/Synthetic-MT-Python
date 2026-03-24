# SyntheticMT Python 重构设计文档

**日期**: 2026-03-24  
**版本**: 1.0  
**状态**: 已批准

---

## 1. 概述

### 1.1 项目背景

SyntheticMT 是一个大地电磁（MT）时间序列合成Python库，基于论文 "Synthesizing magnetotelluric time series based on forward modeling" (Wang P, Chen X, Zhang Y, 2023) 实现。

### 1.2 重构目标

1. **代码结构重构** - 将大的单文件拆分为模块化结构
2. **架构升级** - 采用领域驱动设计（DDD）模式
3. **代码质量** - 改善命名、类型提示、文档
4. **功能重组** - 按功能边界重新组织代码

### 1.3 现有问题

- `synthetic_mt.py` (1466行) 包含过多功能
- 领域逻辑与I/O逻辑混杂
- 缺乏清晰的层次边界
- 部分命名不够清晰

---

## 2. 目标架构

### 2.1 目录结构

```
src/synthetic_mt/
├── domain/                    # 核心业务逻辑（无外部依赖）
│   ├── __init__.py
│   ├── entities.py           # EMFields, ForwardSite 实体
│   ├── value_objects.py      # SyntheticMethod, NoiseType, NoiseConfig
│   └── services/
│       ├── __init__.py
│       ├── synthesis.py     # 时间序列合成服务
│       ├── noise.py          # 噪声注入服务
│       └── calibration.py    # 标定服务
│
├── application/              # 用例层（编排领域对象）
│   ├── __init__.py
│   ├── synthesis_use_case.py  # 合成时间序列用例
│   └── dto.py                # 数据传输对象
│
├── infrastructure/           # 基础设施层（外部依赖）
│   ├── __init__.py
│   ├── io/
│   │   ├── __init__.py
│   │   ├── modem/           # ModEM 文件读写
│   │   ├── phoenix/         # Phoenix TSn/TBL 读写
│   │   └── output/          # GMT/CSV/NumPy 输出
│   └── calibration/
│       ├── __init__.py
│       ├── clb.py            # CLB 标定文件
│       └── clc.py            # CLC 标定文件
│
├── presentation/             # 展示层
│   ├── __init__.py
│   ├── gui.py               # GUI 界面
│   └── cli.py               # 命令行接口
│
└── __init__.py               # 公开API
```

### 2.2 依赖规则

```
presentation/ ──────────────► infrastructure/
        │                           │
        │                           ▼
        └────────────────────► application/
                                   │
                                   ▼
                               domain/
                                   ▲
                                   │
                       (domain 完全独立，无反向依赖)
```

- **domain/** - 完全独立，不依赖其他层
- **infrastructure/** - 依赖 domain/
- **application/** - 依赖 domain/ 和 infrastructure/
- **presentation/** - 依赖所有层

---

## 3. 详细设计

### 3.1 Domain 层

#### 3.1.1 值对象 (value_objects.py)

```python
class SyntheticMethod(Enum):
    """合成方法"""
    FIX = 0                         # 不分段
    FIXED_AVG = 1                   # 固定长度平均分段
    FIXED_AVG_WINDOWED = 2         # 固定长度平均分段+窗函数
    RANDOM_SEG = 3                  # 随机长度分段
    RANDOM_SEG_WINDOWED = 4         # 随机长度分段+窗函数
    RANDOM_SEG_PARTIAL = 5          # 随机长度分段+部分窗 (默认)

class NoiseType(Enum):
    """噪声类型"""
    SQUARE_WAVE = 'square'      # 方波噪声
    TRIANGULAR = 'triangular'   # 三角波噪声
    IMPULSIVE = 'impulsive'    # 冲击噪声
    GAUSSIAN = 'gaussian'      # 高斯噪声
    POWERLINE = 'powerline'     # 工频干扰

@dataclass
class NoiseConfig:
    """噪声配置"""
    noise_type: NoiseType = NoiseType.GAUSSIAN
    amplitude: float = 0.0
    frequency: float = 0.0
    probability: float = 0.01
    phase: float = 0.0
```

#### 3.1.2 实体 (entities.py)

```python
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
    fields: list[EMFields] = field(default_factory=list)
    
    def frequencies(self) -> np.ndarray: ...
    def add_fields(self, f: EMFields): ...
    def update_nature_magnetic_amplitude(self, scale_e, scale_b): ...
    def interpolation(self, per_count: int): ...
    def negative_harmonic_factor(self): ...
    def get_feh1(self) -> tuple: ...
    def add_calibration(self, responds: np.ndarray): ...
```

#### 3.1.3 领域服务

**synthesis.py** - 合成服务（纯算法，无副作用）
- `nature_magnetic_amplitude(freq)` - 自然磁场幅度
- `freq_to_time(amp, phase, freq, sample_rate, n)` - 频率转时间域
- `hanning_window(n)` / `inv_hanning_window(n)` - 窗函数
- `synthesis_service.synthesize()` - 执行合成

**noise.py** - 噪声服务
- `inject_gaussian()` - 高斯噪声
- `inject_square()` - 方波噪声
- `inject_triangular()` - 三角波噪声
- `inject_impulsive()` - 冲击噪声
- `add_powerline()` - 工频干扰

### 3.2 Application 层

```python
@dataclass
class SynthesisRequest:
    """合成请求 DTO"""
    modem_path: str
    ts_config: str
    method: SyntheticMethod
    noise_config: NoiseConfig
    output_format: str

@dataclass
class SynthesisResult:
    """合成结果 DTO"""
    ex: np.ndarray
    ey: np.ndarray
    hx: np.ndarray
    hy: np.ndarray
    hz: np.ndarray
    sample_rate: float
    duration: float
    metadata: dict

class SynthesisUseCase:
    """合成时间序列用例"""
    def execute(self, request: SynthesisRequest) -> SynthesisResult: ...
```

### 3.3 Infrastructure 层

#### 3.3.1 ModEM IO

```
infrastructure/io/modem/
├── __init__.py
├── reader.py      # ModemReader.read()
├── types.py       # ModemFormat, BlockType
└── parser.py      # 内部解析器
```

#### 3.3.2 Phoenix IO

```
infrastructure/io/phoenix/
├── __init__.py
├── tsn.py         # TsnFile.load(), TsnFile.save()
├── tbl.py         # TblFile.load(), TblFile.save()
└── types.py       # TagInfo, PhoenixFormat
```

#### 3.3.3 Output

```
infrastructure/io/output/
├── __init__.py
├── gmt.py         # GmtExporter.export()
├── csv.py         # CsvExporter.export()
└── numpy.py       # NumpyExporter.export/load()
```

### 3.4 Presentation 层

- `gui.py` - PySide6 GUI（保持现有功能）
- `cli.py` - 命令行接口（保持现有功能）

### 3.5 兼容层

```python
# compat.py - 向后兼容适配层
from .domain.entities import ForwardSite
from .domain.value_objects import SyntheticMethod, NoiseType

# 保持原有便捷函数
def load_modem_file(path: str) -> list[ForwardSite]: ...
def create_test_site() -> ForwardSite: ...

SYNTHETIC_METHOD_NAMES = {...}
```

---

## 4. 文件迁移映射

| 现有文件 | 新结构位置 | 变更类型 |
|---------|-----------|---------|
| `synthetic_mt.py` | `domain/entities.py` | 拆分 |
| `synthetic_mt.py` | `domain/value_objects.py` | 拆分 |
| `synthetic_mt.py` | `domain/services/synthesis.py` | 拆分 |
| `synthetic_mt.py` | `domain/services/noise.py` | 拆分 |
| `synthetic_mt.py` | `domain/services/calibration.py` | 拆分 |
| `synthetic_mt.py` | `application/dto.py` | 拆分 |
| `synthetic_mt.py` | `application/synthesis_use_case.py` | 拆分 |
| `synthetic_mt.py` | `infrastructure/io/modem/` | 拆分 |
| `phoenix.py` | `infrastructure/io/phoenix/` | 拆分 |
| `synthetic_mt.py` | `infrastructure/io/output/` | 拆分 |
| `synthetic_mt.py` | `infrastructure/calibration/` | 拆分 |
| `gui.py` | `presentation/gui.py` | 移动 |
| `cli.py` | `presentation/cli.py` | 移动 |
| - | `compat.py` | 新增 |

---

## 5. 实施计划

### Phase 1: 创建目录结构
- 创建所有必需的目录

### Phase 2: 按依赖顺序迁移代码
1. domain/value_objects.py (无依赖)
2. domain/entities.py (无依赖)
3. domain/services/noise.py (依赖1,2)
4. domain/services/synthesis.py (依赖1,2)
5. domain/services/calibration.py
6. application/dto.py
7. application/synthesis_use_case.py
8. infrastructure/io/modem/
9. infrastructure/io/phoenix/
10. infrastructure/io/output/
11. infrastructure/calibration/
12. presentation/gui.py
13. presentation/cli.py

### Phase 3: 创建兼容层
- compat.py
- 更新 __init__.py

### Phase 4: 验证
- 运行现有51个测试
- 确保向后兼容

---

## 6. 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 循环依赖 | 架构破坏 | 严格遵守依赖方向 |
| 迁移引入bug | 功能损坏 | 51个现有测试验证 |
| 破坏向后兼容 | 用户代码报错 | compat.py 适配层 |

---

## 7. 验收标准

1. 所有51个现有测试通过
2. 现有API调用方式保持不变
3. 每个模块有清晰的职责
4. 类型提示完整
5. 文档字符串完整
