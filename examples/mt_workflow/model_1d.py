"""
一维MT地电模型定义

MT1DModel: 层状地电模型
"""

import numpy as np
from typing import List, Tuple, Optional


class MT1DModel:
    """
    一维层状地电模型

    Attributes:
        name: 模型名称
        resistivity: 各层电阻率 (Ohm·m), 最后一层为半空间
        thickness: 各层厚度 (m), 最后一层无厚度(半空间)

    Example:
        # 均匀半空间
        model = MT1DModel('halfspace', resistivity=[100.0])

        # 三层层状模型
        model = MT1DModel('3layer', resistivity=[100, 500, 10], thickness=[100, 200])
    """

    def __init__(
        self, name: str, resistivity: List[float], thickness: List[float] = None
    ):
        self.name = name
        self.resistivity = np.array(resistivity, dtype=float)
        if thickness is None:
            # 均匀半空间
            self.thickness = np.array([], dtype=float)
        else:
            self.thickness = np.array(thickness, dtype=float)

        # 验证
        if len(self.resistivity) < 1:
            raise ValueError("At least one layer required")
        if len(self.thickness) > 0 and len(self.thickness) != len(self.resistivity) - 1:
            raise ValueError("Thickness count must be n_layers - 1")

    @property
    def n_layers(self) -> int:
        """层数"""
        return len(self.resistivity)

    @property
    def is_halfspace(self) -> bool:
        """是否为均匀半空间模型"""
        return len(self.thickness) == 0

    @property
    def is_layered(self) -> bool:
        """是否为层状模型"""
        return len(self.thickness) > 0

    def __repr__(self) -> str:
        if self.is_halfspace:
            return f"MT1DModel('{self.name}', rho={self.resistivity[0]:.1f} Ohm·m, halfspace)"

        layers_str = ", ".join([f"rho={r:.1f}" for r in self.resistivity])
        thicks_str = ", ".join([f"h={t:.1f}" for t in self.thickness])
        return f"MT1DModel('{self.name}', [{layers_str}], [{thicks_str}])"

    def get_layer_params(self, layer_idx: int) -> Tuple[float, float]:
        """
        获取指定层的参数

        Args:
            layer_idx: 层索引 (0-based)

        Returns:
            (resistivity, thickness), 最后一层厚度为0
        """
        rho = self.resistivity[layer_idx]
        h = self.thickness[layer_idx] if layer_idx < len(self.thickness) else 0.0
        return rho, h


def create_uniform_halfspace(name: str, rho: float) -> MT1DModel:
    """创建均匀半空间模型"""
    return MT1DModel(name, resistivity=[rho])


def create_two_layer(name: str, rho1: float, h1: float, rho2: float) -> MT1DModel:
    """创建二层模型"""
    return MT1DModel(name, resistivity=[rho1, rho2], thickness=[h1])


def create_three_layer(
    name: str, rho1: float, h1: float, rho2: float, h2: float, rho3: float
) -> MT1DModel:
    """创建三层模型"""
    return MT1DModel(name, resistivity=[rho1, rho2, rho3], thickness=[h1, h2])


# 预定义的典型模型
PRESET_MODELS = {
    "uniform_100": MT1DModel("uniform_100", resistivity=[100.0]),
    "uniform_1000": MT1DModel("uniform_1000", resistivity=[1000.0]),
    "uniform_10": MT1DModel("uniform_10", resistivity=[10.0]),
    "two_layer_ll": MT1DModel(
        "two_layer_ll", resistivity=[10.0, 1000.0], thickness=[100.0]
    ),
    "two_layer_hl": MT1DModel(
        "two_layer_hl", resistivity=[1000.0, 10.0], thickness=[100.0]
    ),
    "three_layer_hll": MT1DModel(
        "three_layer_hll", resistivity=[1000.0, 10.0, 100.0], thickness=[50.0, 200.0]
    ),
}


def get_preset_model(name: str) -> Optional[MT1DModel]:
    """获取预定义模型"""
    return PRESET_MODELS.get(name)
