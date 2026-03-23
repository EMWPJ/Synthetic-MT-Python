# SyntheticMT - 大地电磁合成时间序列 Python版

基于论文: Wang P, Chen X, Zhang Y (2023) Synthesizing magnetotelluric time series based on forward modeling. Front. Earth Sci. 11:1086749

## 项目结构

```
合成时间序列V2.0/
├── src/
│   ├── __init__.py       # 包初始化
│   ├── synthetic_mt.py    # 核心合成算法
│   └── phoenix.py        # Phoenix格式读写
├── examples/
│   └── test_synthetic.py # 测试示例
├── tests/                 # 测试目录
├── docs/                  # 文档目录
└── README.md
```

## 核心功能

### 1. 时间序列合成 (synthetic_mt.py)

```python
from synthetic_mt import (
    EMFields, Site, TimeSeriesGenerator, 
    MTSchema, SegmentMethod
)

# 定义测点数据
fields = [
    EMFields(
        freq=10.0,
        ex1=complex(1,0), ey1=complex(1,0), hx1=complex(0.01,0),
        hy1=complex(0.01,0), hz1=complex(0,0),
        ex2=complex(1,0), ey2=complex(-1,0), hx2=complex(0.01,0),
        hy2=complex(-0.01,0), hz2=complex(0,0)
    )
]
site = Site(name='Test', x=0, y=0, fields=fields)

# 创建生成器 (TS3模式: 2400Hz采样率)
gen = MTSchema.create('TS3', segment_periods=8)

# 生成10秒数据
ex, ey, hx, hy, hz = gen.generate(10, site, seed=42)
```

### 2. Phoenix格式读写 (phoenix.py)

```python
from phoenix import TsnFile, TblFile

# 读取TSn文件
data, tags = TsnFile.load('data.TS3')

# 保存TSn文件
TsnFile.save('output.TS3', data, tags)

# 读取TBL配置文件
tbl = TblFile('config.TBL')
sample_rate = tbl['HSMP']  # 采样率
```

## 算法原理

### 核心公式

1. **频域→时域**: E(t) = A·cos(2πft + φ)
2. **源模拟**: 两正交偏振源的随机线性组合
3. **分段拼接**: 模拟自然源时变偏振特性

### 分段方法 (SegmentMethod)

| 方法 | 说明 |
|------|------|
| FIXED | 固定长度 |
| FIXED_WINDOWED | 固定长度+Hanning窗 |
| RANDOM | 随机长度 |
| RANDOM_WINDOWED | 随机长度+Hanning窗 |
| RANDOM_PARTIAL | 随机长度+部分窗(默认) |

## 运行测试

```bash
cd 合成时间序列V2.0
python examples/test_synthetic.py
```

## 待完成

- [ ] ModEM正演结果文件解析
- [ ] 仪器响应加载
- [ ] 完整的Phoenix格式支持
- [ ] 命令行界面
- [ ] 可视化界面
