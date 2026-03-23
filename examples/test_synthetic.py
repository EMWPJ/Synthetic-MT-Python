"""
SyntheticMT 测试示例

演示如何使用合成时间序列功能
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, 'src')

from synthetic_mt import (
    EMFields, Site, TimeSeriesGenerator, MTSchema,
    nature_field_amplitude, SegmentMethod
)


def test_basic_synthesis():
    """测试基本合成功能"""
    print("测试1: 基本时间序列合成")
    
    # 创建测试站点数据
    fields = [
        EMFields(
            freq=10.0,
            ex1=complex(1.0, 0), ey1=complex(1.0, 0), hx1=complex(0.01, 0), 
            hy1=complex(0.01, 0), hz1=complex(0, 0),
            ex2=complex(1.0, 0), ey2=complex(-1.0, 0), hx2=complex(0.01, 0),
            hy2=complex(-0.01, 0), hz2=complex(0, 0)
        ),
        EMFields(
            freq=50.0,
            ex1=complex(0.5, 0), ey1=complex(0.5, 0), hx1=complex(0.005, 0),
            hy1=complex(0.005, 0), hz1=complex(0, 0),
            ex2=complex(0.5, 0), ey2=complex(-0.5, 0), hx2=complex(0.005, 0),
            hy2=complex(-0.005, 0), hz2=complex(0, 0)
        ),
    ]
    
    site = Site(name='Test', x=0, y=0, fields=fields)
    
    # 创建生成器
    gen = TimeSeriesGenerator(
        sample_rate=2400,
        freq_min=1,
        freq_max=100,
        segment_periods=8,
        source_scale=1.0,
        method=SegmentMethod.RANDOM_PARTIAL
    )
    
    # 生成10秒数据
    duration = 10
    ex, ey, hx, hy, hz = gen.generate(duration, site, seed=42)
    
    print(f"  采样点数: {len(ex)}")
    print(f"  Ex范围: [{ex.min():.6f}, {ex.max():.6f}]")
    print(f"  Hx范围: [{hx.min():.8f}, {hx.max():.8f}]")
    
    return ex, ey, hx, hy, hz


def test_nature_field():
    """测试自然场幅度模型"""
    print("\n测试2: 自然场幅度模型")
    
    freqs = np.logspace(-4, 4, 9)
    for f in freqs:
        amp = nature_field_amplitude(f)
        print(f"  f={f:10.4e} Hz -> A={amp:.6e} A/m")
    
    return freqs


def test_schemas():
    """测试不同采集模式"""
    print("\n测试3: 不同采集系统配置")
    
    for name in ['TS3', 'TS4', 'TS5']:
        gen = MTSchema.create(name, segment_periods=8)
        print(f"  {name}: 采样率={gen.sample_rate} Hz, "
              f"频率范围=[{gen.freq_min:.2e}, {gen.freq_max:.2e}] Hz")


def test_single_freq():
    """测试单频信号生成"""
    print("\n测试4: 单频信号生成")
    
    from synthetic_mt import single_freq_signal
    
    fs = 2400
    f = 10
    n = 2400  # 1秒
    
    signal = single_freq_signal(1.0, 0, f, fs, n)
    print(f"  频率: {f} Hz, 采样率: {fs} Hz, 采样点: {n}")
    print(f"  信号范围: [{signal.min():.4f}, {signal.max():.4f}]")
    print(f"  理论峰值: ±1.0")


def demo_plot():
    """演示绘图"""
    print("\n测试5: 生成示例图")
    
    # 简单测试数据
    fields = [
        EMFields(
            freq=10.0,
            ex1=complex(1, 0), ey1=complex(1, 0), hx1=complex(0.01, 0),
            hy1=complex(0.01, 0), hz1=complex(0, 0),
            ex2=complex(1, 0), ey2=complex(-1, 0), hx2=complex(0.01, 0),
            hy2=complex(-0.01, 0), hz2=complex(0, 0)
        )
    ]
    
    site = Site(name='Demo', x=0, y=0, fields=fields)
    gen = TimeSeriesGenerator(2400, 1, 100, method=SegmentMethod.RANDOM_PARTIAL)
    
    ex, ey, hx, hy, hz = gen.generate(1.0, site, seed=42)
    
    t = np.arange(len(ex)) / 2400
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    axes[0].plot(t[:2400], ex[:2400], 'b-', lw=0.5)
    axes[0].set_ylabel('Ex (V/m)')
    axes[0].set_title('合成MT时间序列示例 (10 Hz)')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(t[:2400], hx[:2400], 'r-', lw=0.5)
    axes[1].set_xlabel('时间 (s)')
    axes[1].set_ylabel('Hx (A/m)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/demo_timeseries.png', dpi=150)
    print("  已保存图: examples/demo_timeseries.png")
    
    plt.close()
    
    return True


if __name__ == '__main__':
    print("=" * 50)
    print("SyntheticMT 测试套件")
    print("=" * 50)
    
    test_basic_synthesis()
    test_nature_field()
    test_schemas()
    test_single_freq()
    demo_plot()
    
    print("\n" + "=" * 50)
    print("所有测试完成!")
    print("=" * 50)
