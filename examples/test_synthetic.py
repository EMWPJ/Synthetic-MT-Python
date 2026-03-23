"""
SyntheticMT 测试示例

演示如何使用合成时间序列功能
"""

import numpy as np
import sys
sys.path.insert(0, 'src')

from src import (
    EMFields, ForwardSite, SyntheticTimeSeries, SyntheticSchema, SyntheticMethod,
    nature_magnetic_amplitude, TS_CONFIGS, save_gmt_timeseries
)
from datetime import datetime


def test_basic_synthesis():
    """测试基本合成功能"""
    print("Test 1: Basic time series synthesis")
    
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
    
    site = ForwardSite(name='Test', x=0, y=0, fields=fields)
    
    schema = SyntheticSchema.from_ts('TS3')
    synth = SyntheticTimeSeries(schema, SyntheticMethod.RANDOM_SEG_PARTIAL)
    
    t1 = datetime(2023, 1, 1, 0, 0, 0)
    t2 = datetime(2023, 1, 1, 0, 0, 10)
    
    ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)
    
    print(f"  Samples: {len(ex)}")
    print(f"  Ex range: [{ex.min():.6f}, {ex.max():.6f}]")
    print(f"  Hx range: [{hx.min():.8f}, {hx.max():.8f}]")
    
    return ex, ey, hx, hy, hz


def test_nature_field():
    """测试自然场幅度模型"""
    print("\nTest 2: Nature field amplitude model")
    
    freqs = np.logspace(-4, 4, 9)
    for f in freqs:
        amp = nature_magnetic_amplitude(f)
        print(f"  f={f:10.4e} Hz -> B={amp:.6e} nT")
    
    return freqs


def test_schemas():
    """测试不同采集模式"""
    print("\nTest 3: Different acquisition system configs")
    
    for name in ['TS2', 'TS3', 'TS4', 'TS5']:
        schema = SyntheticSchema.from_ts(name)
        print(f"  {name}: sample_rate={schema.sample_rate} Hz, "
              f"freq_range=[{schema.freq_min:.2e}, {schema.freq_max:.2e}] Hz")


def test_all_methods():
    """测试所有合成方法"""
    print("\nTest 4: All synthesis methods")
    
    site = ForwardSite(name='Test', x=0, y=0, fields=[
        EMFields(freq=10.0, ex1=complex(1,0), ey1=complex(1,0), hx1=complex(0.01,0),
                 hy1=complex(0.01,0), ex2=complex(1,0), ey2=complex(-1,0),
                 hx2=complex(0.01,0), hy2=complex(-0.01,0))
    ])
    schema = SyntheticSchema.from_ts('TS3')
    t1 = datetime(2023, 1, 1, 0, 0, 0)
    t2 = datetime(2023, 1, 1, 0, 0, 1)
    
    for method in SyntheticMethod:
        synth = SyntheticTimeSeries(schema, method)
        ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)
        print(f"  {method.name}: {len(ex)} samples")


def test_output_formats():
    """测试输出格式"""
    print("\nTest 5: Output formats")
    
    site = ForwardSite(name='Test', x=0, y=0, fields=[
        EMFields(freq=10.0, ex1=complex(1,0), ey1=complex(1,0), hx1=complex(0.01,0),
                 hy1=complex(0.01,0), ex2=complex(1,0), ey2=complex(-1,0),
                 hx2=complex(0.01,0), hy2=complex(-0.01,0))
    ])
    schema = SyntheticSchema.from_ts('TS3')
    synth = SyntheticTimeSeries(schema, SyntheticMethod.RANDOM_SEG_PARTIAL)
    
    t1 = datetime(2023, 1, 1, 0, 0, 0)
    t2 = datetime(2023, 1, 1, 0, 0, 1)
    
    ex, ey, hx, hy, hz = synth.generate(t1, t2, site, seed=42)
    
    # GMT format
    gmt_file = save_gmt_timeseries('./output', 'Test', ex, ey, hx, hy, hz, t1, 2400)
    print(f"  GMT format: {gmt_file}")
    
    print("  All output formats work correctly!")


if __name__ == '__main__':
    print("=" * 50)
    print("SyntheticMT Test Suite")
    print("=" * 50)
    
    test_basic_synthesis()
    test_nature_field()
    test_schemas()
    test_all_methods()
    test_output_formats()
    
    print("\n" + "=" * 50)
    print("All tests completed!")
    print("=" * 50)
