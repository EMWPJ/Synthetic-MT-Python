"""Domain entities - objects with identity."""

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
    
    def update_nature_magnetic_amplitude(self, scale_e: np.ndarray, scale_b: np.ndarray):
        """
        根据自然磁场强度更新电磁场幅值
        
        Parameters:
            scale_e: 电场缩放因子 (基于Hz幅值)
            scale_b: 磁场缩放因子 (基于Hz幅值)
        """
        for i, f in enumerate(self.fields):
            if i < len(scale_e):
                f.ex1 = complex(abs(f.ex1) * scale_e[i], f.ex1.imag)
                f.ey1 = complex(abs(f.ey1) * scale_e[i], f.ey1.imag)
                f.hx1 = complex(abs(f.hx1) * scale_b[i], f.hx1.imag)
                f.hy1 = complex(abs(f.hy1) * scale_b[i], f.hy1.imag)
                f.hz1 = complex(abs(f.hz1) * scale_b[i], f.hz1.imag)
                f.ex2 = complex(abs(f.ex2) * scale_e[i], f.ex2.imag)
                f.ey2 = complex(abs(f.ey2) * scale_e[i], f.ey2.imag)
                f.hx2 = complex(abs(f.hx2) * scale_b[i], f.hx2.imag)
                f.hy2 = complex(abs(f.hy2) * scale_b[i], f.hy2.imag)
                f.hz2 = complex(abs(f.hz2) * scale_b[i], f.hz2.imag)


def nature_magnetic_amplitude(freq: float) -> float:
    """
    计算自然磁场强度幅值 (单位: nT)
    
    基于MT频率范围和典型自然源磁场强度统计
    磁场强度随频率变化呈幂律分布
    
    Parameters:
        freq: 频率 (Hz)
    
    Returns:
        磁场幅值 (nT)
    
    Note:
        参考: Wang et al. (2023) 论文中的自然磁场模型
        使用分段幂律近似典型MT源的磁场强度谱
    """
    period = 1.0 / freq if freq > 0 else float('inf')
    
    if freq <= 0:
        return 0.0
    
    if freq < 1e-5:
        b = 100.0 * (freq / 1e-5) ** 0.3
    elif freq < 1e-3:
        b = 100.0 * (freq / 1e-5) ** 0.3
    elif freq < 0.01:
        b = 30.0 * (freq / 1e-3) ** 0.2
    elif freq < 0.1:
        b = 20.0 * (freq / 0.01) ** 0.1
    elif freq < 1.0:
        b = 10.0 * (freq / 0.1) ** 0.05
    elif freq < 10.0:
        b = 5.0 * (freq / 1.0) ** -0.1
    elif freq < 100.0:
        b = 3.0 * (freq / 10.0) ** -0.2
    elif freq < 1000.0:
        b = 1.0 * (freq / 100.0) ** -0.3
    else:
        b = 0.3 * (freq / 1000.0) ** -0.4
    
    return max(b, 0.001)
