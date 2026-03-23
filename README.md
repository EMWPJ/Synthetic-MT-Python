# SyntheticMT - 大地电磁合成时间序列 Python版

基于论文: Wang P, Chen X, Zhang Y (2023) Synthesizing magnetotelluric time series based on forward modeling. Front. Earth Sci. 11:1086749

## 项目结构

```
合成时间序列V2.0/
├── src/
│   ├── __init__.py       # 包初始化
│   ├── synthetic_mt.py    # 核心合成算法
│   ├── phoenix.py        # Phoenix格式读写
│   ├── cli.py           # 命令行接口
│   └── gui.py           # PySide6图形界面
├── examples/
│   └── test_synthetic.py # 测试示例
├── tests/                 # 测试目录
├── docs/                  # 文档目录
└── README.md
```

## 核心功能

### 1. 时间序列合成 (synthetic_mt.py)

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

### 2. Phoenix格式读写 (phoenix.py)

```python
from src import TsnFile, TblFile

# 读取TSn文件
data, tags = TsnFile.load('data.TS3')

# 保存TSn文件
TsnFile.save('output.TS3', data, tags)

# 读取TBL配置文件
tbl = TblFile('config.TBL')
sample_rate = tbl['HSMP']  # 采样率
```

### 3. 自然磁场强度计算

```python
from src import nature_magnetic_amplitude, calculate_mt_scale_factors

# 计算给定频率的自然磁场强度
b = nature_magnetic_amplitude(1.0)  # f=1 Hz

# 计算MT缩放因子
scale_e, scale_b = calculate_mt_scale_factors(site)
```

### 4. 噪声注入

```python
from src import NoiseInjector, NoiseConfig, NoiseType

config = NoiseConfig(noise_type=NoiseType.GAUSSIAN, amplitude=0.1)
injector = NoiseInjector(config, sample_rate=2400)

ex_noisy, ey_noisy, hx_noisy, hy_noisy, hz_noisy = injector.add_noise(ex, ey, hx, hy, hz)
```

### 5. 输出格式

```python
from src import save_gmt_timeseries, save_csv_timeseries, save_numpy_timeseries
from datetime import datetime

# GMT格式 (GMT兼容)
save_gmt_timeseries('./output', 'Test', ex, ey, hx, hy, hz, begin_time, 2400)

# CSV格式
save_csv_timeseries('./output/Test.csv', ex, ey, hx, hy, hz)

# NumPy格式
save_numpy_timeseries('./output/Test.npy', ex, ey, hx, hy, hz)
```

### 6. 命令行接口 (cli.py)

```bash
# 生成时间序列
python -m src.cli generate --begin-time "2023-01-01 00:00:00" --end-time "2023-01-01 00:01:00"

# 列出合成方法
python -m src.cli methods

# 列出TS配置
python -m src.cli configs
```

### 7. PySide6图形界面 (gui.py)

```bash
python -m src.gui
```

## 算法原理

### 核心公式

1. **频域→时域**: E(t) = A·cos(2πft + φ)
2. **源模拟**: 两正交偏振源的随机线性组合
3. **分段拼接**: 模拟自然源时变偏振特性

### 分段方法 (SyntheticMethod)

| 方法 | 说明 |
|------|------|
| FIX | 不分段 |
| FIXED_AVG | 固定长度平均分段 |
| FIXED_AVG_WINDOWED | 固定长度平均分段+窗函数 |
| RANDOM_SEG | 随机长度分段 |
| RANDOM_SEG_WINDOWED | 随机长度分段+窗函数 |
| RANDOM_SEG_PARTIAL | 随机长度分段+部分窗 (默认) |

## TS配置

| 配置 | 采样率 | 频率范围 |
|------|--------|----------|
| TS2 | 2400Hz | 10-12000 Hz |
| TS3 | 2400Hz | 1-1000 Hz |
| TS4 | 150Hz | 0.1-10 Hz |
| TS5 | 15Hz | 1e-6-1 Hz |

## 运行方式

### 命令行
```bash
cd 合成时间序列V2.0
python -m src.cli generate --begin-time "2023-01-01 00:00:00" --end-time "2023-01-01 00:01:00"
```

### 图形界面
```bash
cd 合成时间序列V2.0
python -m src.gui
```

### Python脚本
```bash
cd 合成时间序列V2.0
python examples/test_synthetic.py
```

## 已完成功能

- [x] 6种合成方法实现
- [x] ModEM正演结果文件解析
- [x] Phoenix TSn/TBL文件读写
- [x] CLB/CLC标定文件支持
- [x] 命令行界面
- [x] PySide6图形界面
- [x] 噪声注入 (高斯、冲击、工频干扰等)
- [x] 自然磁场强度计算 (NatureMagneticAmplitude)
- [x] GMT格式输出
- [x] CSV/NumPy格式输出
- [x] 系统标定器 (SystemCalibrator)

## 待完成

- [ ] SysCal V7外部程序调用
- [ ] 可视化绘图
- [ ] 更多格式支持
