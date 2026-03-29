"""
一维MT正演计算

使用递归算法计算层状地电模型的MT响应

算法:
1. 对于均匀半空间，直接计算解析解
2. 对于层状模型，使用递推算法从底向上计算表面阻抗
"""

import numpy as np
from typing import Dict, Tuple
from .model_1d import MT1DModel
from .config import MU0


class MT1DForward:
    """
    一维MT正演计算器

    使用递归算法计算层状模型的阻抗响应

    参考:
    - Wait, J.R. (1954) On the relation between telluric and magnetotelluric fields
    - Cagniard, L. (1953) Basic theory of the magneto-telluric method
    """

    def __init__(self, model: MT1DModel):
        self.model = model

    def calculate_impedance(self, periods: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算1D模型的阻抗张量

        Args:
            periods: 周期数组 (s)

        Returns:
            包含 Zxx, Zxy, Zyx, Zyy 的字典
            阻抗单位: V/(A·m) = Ohm·m (S.I.)
        """
        n = len(periods)
        zxx = np.zeros(n, dtype=complex)
        zxy = np.zeros(n, dtype=complex)
        zyx = np.zeros(n, dtype=complex)
        zyy = np.zeros(n, dtype=complex)

        for i, T in enumerate(periods):
            omega = 2 * np.pi / T

            # 计算表面阻抗 Zxy (TE模式)
            zxy[i] = self._compute_surface_impedance(omega)
            # 1D模型: Zxx = Zyy = 0, Zyx = -Zxy
            zxx[i] = complex(0, 0)
            zyx[i] = -zxy[i]
            zyy[i] = complex(0, 0)

        return {"Zxx": zxx, "Zxy": zxy, "Zyx": zyx, "Zyy": zyy}

    def _compute_surface_impedance(self, omega: float) -> complex:
        """
        计算表面阻抗 (TE模式)

        使用递推算法从最后一层向第一层计算

        标准MT层状模型递推算法 (Wait, 1954; Cagniard, 1953):
        - 从最底层(半空间)开始计算本征阻抗
        - 向上递推: Z_{i-1} = Z_i * (Z_n + Z_i * tanh(k_i*h_i)) / (Z_i + Z_n * tanh(k_i*h_i))
        其中 Z_i 是当前层本征阻抗, Z_n 是下来方向的负载阻抗(来自下层)

        Args:
            omega: 角频率 (rad/s)

        Returns:
            表面阻抗 Zxy (complex)
        """
        if self.model.is_halfspace:
            # 均匀半空间解析解
            return self._halfspace_impedance(omega, self.model.resistivity[0])

        # 层状模型递推算法
        n_layers = self.model.n_layers

        # 最底层是半空间，其本征阻抗
        # Z_n = (1+i) * sqrt(omega * mu0 * rho_n / 2)
        rho_basement = self.model.resistivity[-1]
        z_down = (1 + 1j) * np.sqrt(omega * MU0 * rho_basement / 2)

        # 递推阻抗，从倒数第二层向上到第一层
        for layer_idx in range(n_layers - 2, -1, -1):
            rho, h = self.model.get_layer_params(layer_idx)

            # 本层的本征阻抗
            z_i = (1 + 1j) * np.sqrt(omega * MU0 * rho / 2)

            # 波数 k = sqrt(i * omega * mu0 / rho)
            k = np.sqrt(1j * omega * MU0 / rho)

            # 递推公式 (传输线 analog):
            # Z_up = z_i * (z_down + z_i * tanh(k*h)) / (z_i + z_down * tanh(k*h))
            # z_i: 当前层本征阻抗, z_down: 下来方向的负载阻抗
            if np.abs(k * h) < 100:
                tanh_kh = np.tanh(k * h)
            else:
                tanh_kh = 1.0 + 1e-10  # 避免数值溢出

            numerator = z_down + z_i * tanh_kh
            denominator = z_i + z_down * tanh_kh

            if np.abs(denominator) > 1e-20:
                z_up = z_i * numerator / denominator
            else:
                z_up = z_i

            z_down = z_up  # 继续向上递推

        return z_up

    def _halfspace_impedance(self, omega: float, rho: float) -> complex:
        """
        均匀半空间阻抗解析解

        Z = sqrt(omega * mu0 / sigma) * (1 - i) / sqrt(2)
          = (1 + i) * sqrt(omega * mu0 * rho / 2)

        Args:
            omega: 角频率
            rho: 电阻率 (Ohm·m)

        Returns:
            阻抗 (Ohm·m)
        """
        # 皮肤深度 delta = sqrt(2 * rho / (omega * mu0))
        # 阻抗 Z = (1 - i) * rho / (delta * sqrt(2)) = (1 + i) * sqrt(omega * mu0 * rho / 2)
        return (1 + 1j) * np.sqrt(omega * MU0 * rho / 2)

    def _wave_number(self, omega: float, rho: float) -> complex:
        """
        计算波数 k = sqrt(i * omega * mu0 / rho)

        Args:
            omega: 角频率
            rho: 电阻率

        Returns:
            波数 k (complex)
        """
        # k = (1 + i) / delta, where delta = sqrt(2 * rho / (omega * mu0))
        # k = sqrt(i * omega * mu0 / rho)
        return np.sqrt(1j * omega * MU0 / rho)

    def _reflection_coefficient(
        self, k: complex, z_load: complex, omega: float
    ) -> complex:
        """
        计算反射系数

        r = (z_load - z_wave) / (z_load + z_wave)
        where z_wave = omega * mu0 / k
        """
        z_wave = omega * MU0 / k
        return (z_load - z_wave) / (z_load + z_wave)

    def calculate_app_resistivity_phase(
        self, periods: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算视电阻率和相位

        视电阻率: rho_a = |Zxy|^2 / (omega * mu0)  [Ohm·m]
        相位: phi = arctan(Im(Zxy) / Re(Zxy))  [degrees]

        Args:
            periods: 周期数组 (s)

        Returns:
            (rho_a, phase): 视电阻率数组和相位数组(度)
        """
        periods = np.asarray(periods)  # Ensure numpy array
        impedance = self.calculate_impedance(periods)
        zxy = impedance["Zxy"]

        # 视电阻率 (SI单位)
        omega = 2 * np.pi / periods
        rho_a = np.abs(zxy) ** 2 / (omega * MU0)

        # 相位
        phase = np.arctan2(zxy.imag, zxy.real) * 180.0 / np.pi

        return rho_a, phase

    def calculate_fields(self, periods: np.ndarray) -> "list[EMFields]":
        """
        计算两个极化模式的4分量电磁场

        Args:
            periods: 周期数组 (s)

        Returns:
            EMFields列表
        """
        # 延迟导入避免循环依赖
        from synthetic_mt import EMFields, nature_magnetic_amplitude

        impedance = self.calculate_impedance(periods)
        fields = []

        for i, T in enumerate(periods):
            f = 1.0 / T
            omega = 2 * np.pi / T

            # 参考磁场幅度
            B_ref = nature_magnetic_amplitude(f)
            H_ref = B_ref * 1e-9 / MU0  # nT -> A/m

            zxy = impedance["Zxy"][i]
            zyx = impedance["Zyx"][i]

            # ================================================================
            # 极化1: TM模式 (Ey大, Hx小)
            # TM: Hy驱动, Ex响应
            # ================================================================
            # 归一化的磁场
            hx1 = H_ref * 0.01  # 很小的Hx分量
            hy1 = H_ref  # 主要Hy分量

            # 电场: Ex = Zxy * Hy
            ex1 = zxy * hy1
            # Ey ≈ 0 (理想情况下)
            ey1 = zyx * hx1

            hz1 = complex(0, 0)

            # ================================================================
            # 极化2: TE模式 (Ex大, Hy小)
            # TE: Hx驱动, Ey响应
            # ================================================================
            hx2 = H_ref  # 主要Hx分量
            hy2 = H_ref * 0.01  # 很小的Hy分量

            # Ey = Zyx * Hx
            ey2 = zyx * hx2
            # Ex ≈ 0
            ex2 = zxy * hy2

            hz2 = complex(0, 0)

            fields.append(
                EMFields(
                    freq=f,
                    # 极化1 (TM)
                    ex1=complex(ex1.real, ex1.imag),
                    ey1=complex(ey1.real, ey1.imag),
                    hx1=complex(hx1, 0),
                    hy1=complex(hy1, 0),
                    hz1=hz1,
                    # 极化2 (TE)
                    ex2=complex(ex2.real, ex2.imag),
                    ey2=complex(ey2.real, ey2.imag),
                    hx2=complex(hx2, 0),
                    hy2=complex(hy2, 0),
                    hz2=hz2,
                    # 阻抗张量
                    zxx=impedance["Zxx"][i],
                    zxy=impedance["Zxy"][i],
                    zyx=impedance["Zyx"][i],
                    zyy=impedance["Zyy"][i],
                    # Tipper
                    tzx=complex(0, 0),
                    tzy=complex(0, 0),
                )
            )

        return fields


def compute_theoretical_response(model: MT1DModel, periods: np.ndarray) -> Dict:
    """
    计算理论响应 (便捷函数)

    Args:
        model: 1D模型
        periods: 周期数组

    Returns:
        包含阻抗、视电阻率、相位的字典
    """
    forward = MT1DForward(model)
    impedance = forward.calculate_impedance(periods)
    rho_a, phase = forward.calculate_app_resistivity_phase(periods)

    return {
        "periods": periods,
        "frequencies": 1.0 / periods,
        "impedance": impedance,
        "app_resistivity": rho_a,
        "phase": phase,
    }
