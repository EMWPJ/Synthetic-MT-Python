"""
后台算法核心模块

整合所有MT算法模块，提供独立的计算逻辑。
此模块不依赖GUI，可独立运行和测试。
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# 物理常数
MU0 = 4 * np.pi * 1e-7  # 真空磁导率 (H/m)


# ============================================================================
# 1D 模型定义
# ============================================================================


class MT1DModel:
    """一维层状地电模型"""

    def __init__(
        self, name: str, resistivity: List[float], thickness: List[float] = None
    ):
        self.name = name
        self.resistivity = np.array(resistivity, dtype=float)
        if thickness is None:
            self.thickness = np.array([], dtype=float)
        else:
            self.thickness = np.array(thickness, dtype=float)

    @property
    def n_layers(self) -> int:
        return len(self.resistivity)

    @property
    def is_halfspace(self) -> bool:
        return len(self.thickness) == 0

    def get_layer_params(self, layer_idx: int) -> Tuple[float, float]:
        rho = self.resistivity[layer_idx]
        h = self.thickness[layer_idx] if layer_idx < len(self.thickness) else 0.0
        return rho, h

    def __repr__(self) -> str:
        if self.is_halfspace:
            return f"MT1DModel('{self.name}', rho={self.resistivity[0]:.1f} Ohm·m)"
        layers = ", ".join([f"rho={r:.1f}" for r in self.resistivity])
        thicks = ", ".join([f"h={t:.1f}" for t in self.thickness])
        return f"MT1DModel('{self.name}', [{layers}], [{thicks}])"


# ============================================================================
# 1D 正演计算
# ============================================================================


class MT1DForward:
    """一维MT正演计算器"""

    def __init__(self, model: MT1DModel):
        self.model = model

    def calculate_impedance(self, periods: np.ndarray) -> Dict[str, np.ndarray]:
        """计算阻抗张量"""
        n = len(periods)
        zxx = np.zeros(n, dtype=complex)
        zxy = np.zeros(n, dtype=complex)
        zyx = np.zeros(n, dtype=complex)
        zyy = np.zeros(n, dtype=complex)

        for i, T in enumerate(periods):
            omega = 2 * np.pi / T
            zxy[i] = self._compute_surface_impedance(omega)
            zxx[i] = complex(0, 0)
            zyx[i] = -zxy[i]
            zyy[i] = complex(0, 0)

        return {"Zxx": zxx, "Zxy": zxy, "Zyx": zyx, "Zyy": zyy}

    def _compute_surface_impedance(self, omega: float) -> complex:
        """计算表面阻抗 (递推算法)"""
        if self.model.is_halfspace:
            return self._halfspace_impedance(omega, self.model.resistivity[0])

        n_layers = self.model.n_layers
        rho_basement = self.model.resistivity[-1]
        z_down = (1 + 1j) * np.sqrt(omega * MU0 * rho_basement / 2)

        for layer_idx in range(n_layers - 2, -1, -1):
            rho, h = self.model.get_layer_params(layer_idx)
            z_i = (1 + 1j) * np.sqrt(omega * MU0 * rho / 2)
            k = np.sqrt(1j * omega * MU0 / rho)

            if np.abs(k * h) < 100:
                tanh_kh = np.tanh(k * h)
            else:
                tanh_kh = 1.0 + 1e-10

            numerator = z_down + z_i * tanh_kh
            denominator = z_i + z_down * tanh_kh

            if np.abs(denominator) > 1e-20:
                z_up = z_i * numerator / denominator
            else:
                z_up = z_i

            z_down = z_up

        return z_down  # 修复: 返回 z_down 不是 z_up

    def _halfspace_impedance(self, omega: float, rho: float) -> complex:
        """均匀半空间解析解"""
        return (1 + 1j) * np.sqrt(omega * MU0 * rho / 2)

    def calculate_app_resistivity_phase(
        self, periods: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """计算视电阻率和相位"""
        periods = np.asarray(periods)
        impedance = self.calculate_impedance(periods)
        zxy = impedance["Zxy"]

        omega = 2 * np.pi / periods
        rho_a = np.abs(zxy) ** 2 / (omega * MU0)
        phase = np.arctan2(zxy.imag, zxy.real) * 180.0 / np.pi

        return rho_a, phase

    def calculate_fields(self, periods: np.ndarray) -> List:
        """计算两个极化模式的4分量电磁场"""
        from synthetic_mt import EMFields, nature_magnetic_amplitude

        impedance = self.calculate_impedance(periods)
        fields = []

        for i, T in enumerate(periods):
            f = 1.0 / T
            omega = 2 * np.pi / T

            B_ref = nature_magnetic_amplitude(f)
            H_ref = B_ref * 1e-9 / MU0

            zxy = impedance["Zxy"][i]
            zyx = impedance["Zyx"][i]

            # TM模式
            hx1 = H_ref * 0.01
            hy1 = H_ref
            ex1 = zxy * hy1
            ey1 = zyx * hx1
            hz1 = complex(0, 0)

            # TE模式
            hx2 = H_ref
            hy2 = H_ref * 0.01
            ey2 = zyx * hx2
            ex2 = zxy * hy2
            hz2 = complex(0, 0)

            fields.append(
                EMFields(
                    freq=f,
                    ex1=complex(ex1.real, ex1.imag),
                    ey1=complex(ey1.real, ey1.imag),
                    hx1=complex(hx1, 0),
                    hy1=complex(hy1, 0),
                    hz1=hz1,
                    ex2=complex(ex2.real, ex2.imag),
                    ey2=complex(ey2.real, ey2.imag),
                    hx2=complex(hx2, 0),
                    hy2=complex(hy2, 0),
                    hz2=hz2,
                    zxx=impedance["Zxx"][i],
                    zxy=impedance["Zxy"][i],
                    zyx=impedance["Zyx"][i],
                    zyy=impedance["Zyy"][i],
                    tzx=complex(0, 0),
                    tzy=complex(0, 0),
                )
            )

        return fields


# ============================================================================
# 时间序列合成
# ============================================================================


class TimeSeriesSynthesizer:
    """MT时间序列合成器"""

    def __init__(self, config: "TSConfig"):
        from synthetic_mt import SyntheticSchema, SyntheticTimeSeries, SyntheticMethod

        self.config = config
        self.schema = SyntheticSchema(
            name=config.name,
            sample_rate=config.sample_rate,
            freq_min=config.freq_min,
            freq_max=config.freq_max,
        )
        self.synth = SyntheticTimeSeries(
            self.schema, SyntheticMethod.RANDOM_SEG_PARTIAL
        )

    def generate(
        self, site, t1: datetime, t2: datetime, seed: Optional[int] = None
    ) -> Tuple[np.ndarray, ...]:
        """生成时间序列"""
        return self.synth.generate(t1, t2, site, seed=seed)

    def generate_duration(
        self,
        site,
        duration: float,
        start_time: Optional[datetime] = None,
        seed: Optional[int] = None,
    ) -> Tuple:
        """按指定时长生成"""
        if start_time is None:
            start_time = datetime(2023, 1, 1, 0, 0, 0)
        t1 = start_time
        t2 = datetime(2023, 1, 1, 0, 0, int(duration))
        ex, ey, hx, hy, hz = self.generate(site, t1, t2, seed=seed)
        return ex, ey, hx, hy, hz, t1, t2


# ============================================================================
# 时间序列处理
# ============================================================================


class SpectrumResult:
    """频谱分析结果"""

    def __init__(
        self,
        frequencies: np.ndarray,
        amplitude: np.ndarray,
        phase: np.ndarray,
        power: np.ndarray,
        n_samples: int,
        sample_rate: float,
    ):
        self.frequencies = frequencies
        self.amplitude = amplitude
        self.phase = phase
        self.power = power
        self.n_samples = n_samples
        self.sample_rate = sample_rate


class TimeSeriesProcessor:
    """时间序列处理器"""

    def __init__(
        self,
        ex: np.ndarray,
        ey: np.ndarray,
        hx: np.ndarray,
        hy: np.ndarray,
        hz: np.ndarray,
        sample_rate: float,
    ):
        self.ex = ex
        self.ey = ey
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.sample_rate = sample_rate
        self.n = len(ex)
        self.duration = self.n / sample_rate
        self._freqs = np.fft.fftfreq(self.n, 1.0 / sample_rate)

    def compute_fft(self, signal: np.ndarray, remove_dc: bool = True) -> SpectrumResult:
        """计算FFT"""
        if remove_dc:
            signal = signal - np.mean(signal)

        fft_vals = np.fft.fft(signal)
        pos_mask = self._freqs > 0
        freqs = self._freqs[pos_mask]
        fft_pos = fft_vals[pos_mask]

        amplitude = np.abs(fft_pos) * 2.0 / self.n
        phase = np.angle(fft_pos, deg=True)
        power = amplitude**2

        return SpectrumResult(freqs, amplitude, phase, power, self.n, self.sample_rate)

    def estimate_impedance_at_periods(
        self, periods: np.ndarray, freq_tol: float = 0.1
    ) -> Dict:
        """在指定周期处估算阻抗（使用全张量最小二乘法）"""
        return self._estimate_impedance_ls(periods, freq_tol)

    def _estimate_impedance_ls(
        self, periods: np.ndarray, freq_tol: float = 0.1
    ) -> Dict:
        """
        使用全张量阻抗最小二乘法估算阻抗

        理论基础:
        E(ω) = Z(ω) * H(ω) + ε(ω)

        对于2x2张量:
        [Ex]   [Zxx Zxy] [Hx]   [εx]
        [Ey] = [Zyx Zyy]*[Hy] + [εy]

        最小二乘解: Z = C_EH * C_HH^(-1)

        其中:
        - C_EH = <E * H^H> 是互协方差矩阵 (2x2)
        - C_HH = <H * H^H> 是自协方差矩阵 (2x2)

        注意: 对于1D MT模型，阻抗张量有如下约束:
        - Zxx ≈ 0, Zyy ≈ 0 (对角元为零)
        - Zyx = -Zxy (反对称)

        由于磁场Hx和Hy通常高度相关，直接求逆C_HH会因病态矩阵而失败。
        因此使用正则化或奇异值分解(SVD)来稳定求解。
        """
        # 计算各分量FFT (去除直流)
        ex_fft = np.fft.fft(self.ex - np.mean(self.ex))
        ey_fft = np.fft.fft(self.ey - np.mean(self.ey))
        hx_fft = np.fft.fft(self.hx - np.mean(self.hx))
        hy_fft = np.fft.fft(self.hy - np.mean(self.hy))

        pos_mask = self._freqs > 0
        freqs = self._freqs[pos_mask]

        # 归一化因子
        norm = 2.0 / self.n

        # 计算所有交叉谱
        ExHx = (ex_fft * np.conj(hx_fft))[pos_mask] * norm
        ExHy = (ex_fft * np.conj(hy_fft))[pos_mask] * norm
        EyHx = (ey_fft * np.conj(hx_fft))[pos_mask] * norm
        EyHy = (ey_fft * np.conj(hy_fft))[pos_mask] * norm

        HxHx = (hx_fft * np.conj(hx_fft))[pos_mask] * norm
        HyHy = (hy_fft * np.conj(hy_fft))[pos_mask] * norm
        HxHy = (hx_fft * np.conj(hy_fft))[pos_mask] * norm
        HyHx = (hy_fft * np.conj(hx_fft))[pos_mask] * norm

        ExEx = (ex_fft * np.conj(ex_fft))[pos_mask] * norm
        EyEy = (ey_fft * np.conj(ey_fft))[pos_mask] * norm
        ExEy = (ex_fft * np.conj(ey_fft))[pos_mask] * norm

        # 张量阻抗最小二乘解: Z = C_EH * C_HH^(-1)
        n_freqs = len(freqs)

        Zxx = np.zeros(n_freqs, dtype=complex)
        Zxy = np.zeros(n_freqs, dtype=complex)
        Zyx = np.zeros(n_freqs, dtype=complex)
        Zyy = np.zeros(n_freqs, dtype=complex)

        # 残差和相干性（用于质量控制）
        residual = np.zeros(n_freqs, dtype=complex)
        coherence = np.zeros(n_freqs, dtype=float)

        eps = 1e-20
        # 正则化参数，用于稳定矩阵求逆
        reg = 1e-10

        for i in range(n_freqs):
            # C_HH = [[HxHx, HxHy], [HyHx, HyHy]]
            # C_EH = [[ExHx, ExHy], [EyHx, EyHy]]
            C_HH = np.array([[HxHx[i], HxHy[i]], [HyHx[i], HyHy[i]]])
            C_EH = np.array([[ExHx[i], ExHy[i]], [EyHx[i], EyHy[i]]])

            # 使用正则化稳定矩阵求逆: C_HH_reg = C_HH + reg*I
            C_HH_reg = C_HH + reg * np.eye(2)
            det = C_HH_reg[0, 0] * C_HH_reg[1, 1] - C_HH_reg[0, 1] * C_HH_reg[1, 0]

            if np.abs(det) > eps:
                # C_HH^(-1) = [[d, -b], [-c, a]] / det where C_HH = [[a, b], [c, d]]
                C_HH_inv = (
                    np.array(
                        [
                            [C_HH_reg[1, 1], -C_HH_reg[0, 1]],
                            [-C_HH_reg[1, 0], C_HH_reg[0, 0]],
                        ]
                    )
                    / det
                )

                Z_tensor = C_EH @ C_HH_inv

                Zxx[i] = Z_tensor[0, 0]
                Zxy[i] = Z_tensor[0, 1]
                Zyx[i] = Z_tensor[1, 0]
                Zyy[i] = Z_tensor[1, 1]

                # 计算拟合残差
                E_fit = C_EH @ C_HH_inv @ C_HH
                residual[i] = np.mean(np.abs(C_EH - E_fit))

                # 计算相干性: Coh = |C_EH|^2 / (C_EE * C_HH)
                C_EE = np.array([[ExEx[i], ExEy[i]], [np.conj(ExEy[i]), EyEy[i]]])
                C_EE_trace = np.abs(ExEx[i]) + np.abs(EyEy[i])
                C_HH_trace = np.abs(HxHx[i]) + np.abs(HyHy[i])
                C_EH_fro = np.sqrt(
                    np.abs(ExHx[i]) ** 2
                    + np.abs(ExHy[i]) ** 2
                    + np.abs(EyHx[i]) ** 2
                    + np.abs(EyHy[i]) ** 2
                )
                coherence[i] = (C_EH_fro**2) / (C_EE_trace * C_HH_trace + eps)
            else:
                # C_HH奇异，使用备选方法
                # 对于1D: Zxy = ExHy/HyHy (简单比值法)
                Zxy[i] = ExHy[i] / (HyHy[i] + eps)
                Zyx[i] = -Zxy[i]  # 1D约束
                Zxx[i] = 0
                Zyy[i] = 0

        # 插值到目标周期
        target_freqs = 1.0 / periods
        n = len(periods)
        zxx_est = np.zeros(n, dtype=complex)
        zxy_est = np.zeros(n, dtype=complex)
        zyx_est = np.zeros(n, dtype=complex)
        zyy_est = np.zeros(n, dtype=complex)

        for i, target_f in enumerate(target_freqs):
            idx = np.argmin(np.abs(freqs - target_f))
            if np.abs(freqs[idx] - target_f) / target_f < freq_tol:
                zxx_est[i] = Zxx[idx]
                zxy_est[i] = Zxy[idx]
                zyx_est[i] = Zyx[idx]
                zyy_est[i] = Zyy[idx]

        # 计算视电阻率和相位（使用Zxy）
        omega = 2 * np.pi / periods
        rho_a = np.abs(zxy_est) ** 2 / (omega * MU0)
        phase = np.arctan2(zxy_est.imag, zxy_est.real) * 180.0 / np.pi

        return {
            "periods": periods,
            "frequencies": target_freqs,
            "Zxx": zxx_est,
            "Zxy": zxy_est,
            "Zyx": zyx_est,
            "Zyy": zyy_est,
            "app_resistivity": rho_a,
            "phase": phase,
            "coherence": coherence,
            "residual": residual,
        }

    def estimate_impedance_simple(
        self, periods: np.ndarray, freq_tol: float = 0.1
    ) -> Dict:
        """
        使用简单比值法估算阻抗（旧方法，用于对比）

        Zxy = <ExHy> / <HyHy>
        Zyx = <EyHx> / <HxHx>
        """
        # 计算各分量FFT
        ex_fft = np.fft.fft(self.ex - np.mean(self.ex))
        ey_fft = np.fft.fft(self.ey - np.mean(self.ey))
        hx_fft = np.fft.fft(self.hx - np.mean(self.hx))
        hy_fft = np.fft.fft(self.hy - np.mean(self.hy))

        pos_mask = self._freqs > 0
        freqs = self._freqs[pos_mask]

        ExHy = (ex_fft * np.conj(hy_fft))[pos_mask] * 2.0 / self.n
        EyHx = (ey_fft * np.conj(hx_fft))[pos_mask] * 2.0 / self.n
        HyHy = (hy_fft * np.conj(hy_fft))[pos_mask] * 2.0 / self.n
        HxHx = (hx_fft * np.conj(hx_fft))[pos_mask] * 2.0 / self.n

        eps = 1e-20
        Zxy = ExHy / (HyHy + eps)
        Zyx = EyHx / (HxHx + eps)

        target_freqs = 1.0 / periods
        n = len(periods)
        zxy_est = np.zeros(n, dtype=complex)
        zyx_est = np.zeros(n, dtype=complex)

        for i, target_f in enumerate(target_freqs):
            idx = np.argmin(np.abs(freqs - target_f))
            if np.abs(freqs[idx] - target_f) / target_f < freq_tol:
                zxy_est[i] = Zxy[idx]
                zyx_est[i] = Zyx[idx]

        omega = 2 * np.pi / periods
        rho_a = np.abs(zxy_est) ** 2 / (omega * MU0)
        phase = np.arctan2(zxy_est.imag, zxy_est.real) * 180.0 / np.pi

        return {
            "periods": periods,
            "frequencies": target_freqs,
            "Zxy": zxy_est,
            "Zyx": zyx_est,
            "app_resistivity": rho_a,
            "phase": phase,
        }


# ============================================================================
# 验证器
# ============================================================================


class Model1DValidator:
    """1D模型特征验证器"""

    def __init__(self, fields: List = None):
        self.fields = fields

    def check_impedance_symmetry(self) -> Dict:
        """检查阻抗反对称性: Zxy = -Zyx"""
        if self.fields is None:
            return {"passed": False, "message": "No fields provided"}

        zxy_mag = np.mean([abs(f.zxy) for f in self.fields])
        zyx_mag = np.mean([abs(f.zyx) for f in self.fields])

        if zyx_mag < 1e-20:
            ratio = 1.0
        else:
            ratio = zxy_mag / zyx_mag

        tolerance = 0.05
        passed = abs(ratio - 1.0) < tolerance

        return {
            "passed": passed,
            "name": "impedance_symmetry",
            "message": f"Zxy/Zyx = {ratio:.4f}",
            "value": ratio,
            "tolerance": tolerance,
        }

    def check_zero_diagonal(self) -> Dict:
        """检查阻抗对角元为零"""
        if self.fields is None:
            return {"passed": False, "message": "No fields provided"}

        zxx_max = max([abs(f.zxx) for f in self.fields]) if self.fields else 0
        zyy_max = max([abs(f.zyy) for f in self.fields]) if self.fields else 0

        max_val = max(zxx_max, zyy_max)
        tolerance = 1e-10
        passed = max_val < tolerance

        return {
            "passed": passed,
            "name": "zero_diagonal",
            "message": f"Zxx_max={zxx_max:.2e}, Zyy_max={zyy_max:.2e}",
            "value": max_val,
            "tolerance": tolerance,
        }

    def validate_all(self) -> List[Dict]:
        """执行所有验证"""
        return [
            self.check_impedance_symmetry(),
            self.check_zero_diagonal(),
        ]


class ResultsComparator:
    """正演与反演结果对比器"""

    def __init__(
        self,
        periods: np.ndarray,
        forward_rho: np.ndarray,
        forward_phase: np.ndarray,
        estimated_rho: np.ndarray = None,
        estimated_phase: np.ndarray = None,
    ):
        self.periods = periods
        self.forward_rho = forward_rho
        self.forward_phase = forward_phase
        self.estimated_rho = estimated_rho
        self.estimated_phase = estimated_phase

    def compute_rho_error(self) -> np.ndarray:
        """计算视电阻率相对误差"""
        if self.estimated_rho is None:
            return None
        with np.errstate(divide="ignore", invalid="ignore"):
            error = (
                np.abs(self.forward_rho - self.estimated_rho) / self.forward_rho * 100
            )
            error = np.nan_to_num(error, nan=0, posinf=0, neginf=0)
        return error

    def compute_phase_error(self) -> np.ndarray:
        """计算相位绝对误差"""
        if self.estimated_phase is None:
            return None
        return np.abs(self.forward_phase - self.estimated_phase)


# ============================================================================
# 配置
# ============================================================================


class TSConfig:
    """采集系统配置"""

    def __init__(
        self,
        name: str,
        sample_rate: float,
        freq_min: float,
        freq_max: float,
        period_min: float,
        period_max: float,
        description: str = "",
    ):
        self.name = name
        self.sample_rate = sample_rate
        self.freq_min = freq_min
        self.freq_max = freq_max
        self.period_min = period_min
        self.period_max = period_max
        self.description = description

    def __repr__(self) -> str:
        return f"TSConfig('{self.name}', {self.sample_rate}Hz)"


# Phoenix采集系统配置
TS_CONFIGS = {
    "TS3": TSConfig("TS3", 2400, 1, 1000, 1 / 1000, 1 / 1, "Broadband 2400Hz"),
    "TS4": TSConfig("TS4", 150, 0.1, 10, 0.1, 10, "Short period 150Hz"),
    "TS5": TSConfig("TS5", 15, 1e-6, 1, 1, 1e6, "Long period 15Hz"),
}


def get_default_forward_periods() -> np.ndarray:
    """获取默认正演周期数组"""
    periods = np.logspace(-3, 5, 161)  # 1ms ~ 100000s
    return periods


def get_default_processing_periods() -> np.ndarray:
    """获取默认处理周期数组"""
    periods = np.logspace(-2.5, 4.3, 100)  # ~3ms ~ 20000s
    return periods


# ============================================================================
# 确定性时间序列合成器
# ============================================================================


# ============================================================================
# 随机分段MT时间序列合成器 (优化版)
# ============================================================================


class RandomSegmentTimeSeriesSynthesizer:
    """
    优化的随机分段MT时间序列合成器

    基于论文 Wang et al. (2023) 的 RANDOM_SEG_PARTIAL 算法实现，包含以下特性：

    1. 随机振幅: TM模式用高斯N(1,1), TE模式用均匀[0,2]
    2. 随机相位: 每段均匀随机[0, 2π)
    3. 随机分段长度: 高斯分布，均值为synthetic_periods个周期
    4. 边界余弦窗: 每段边界应用余弦渐变窗保证连续性
    5. 每段极化角: θ ∈ [0, 2π)，控制TE/TM混合比例
    6. 单频分段拼接 + 多频叠加

    原理:
    - 每个频点独立生成分段时间序列
    - 每段随机极化角: ex = ex_TM * cos(θ) + ex_TE * sin(θ)
    - 边界窗保证相邻段之间平滑过渡
    - 最后所有频点时域信号叠加
    """

    def __init__(
        self,
        sample_rate: float = 2400,
        synthetic_periods: float = 8.0,
        source_scale: float = 1.0,
    ):
        """
        初始化合成器

        Args:
            sample_rate: 采样率 (Hz)
            synthetic_periods: 合成周期数 (用于计算平均段长度)
            source_scale: 源幅值缩放因子
        """
        self.sample_rate = sample_rate
        self.synthetic_periods = synthetic_periods
        self.source_scale = source_scale

    def generate_from_fields(
        self,
        fields: List,
        duration: float = 10.0,
        seed: int = None,
        start_time: float = 0.0,
    ) -> Dict:
        """
        从EMFields列表生成时间序列

        Args:
            fields: EMFields列表 (包含ex1, ey1, hx1, hy1, hz1, ex2, ey2, hx2, hy2, hz2等)
            duration: 时长 (秒)
            seed: 随机种子
            start_time: 开始时间 (秒), 用于计算 deltaPha

        Returns:
            包含时间序列的字典: {"ex": array, "ey": array, "hx": array, "hy": array, "hz": array, ...}
        """
        rng = np.random.default_rng(seed)
        n_samples = int(self.sample_rate * duration)

        # 初始化输出数组
        ex = np.zeros(n_samples)
        ey = np.zeros(n_samples)
        hx = np.zeros(n_samples)
        hy = np.zeros(n_samples)
        hz = np.zeros(n_samples)

        # 时间偏移 (用于 deltaPha 计算)
        deltatime = start_time  # 秒

        for f in fields:
            freq = f.freq
            if freq <= 0:
                continue

            # 计算 deltaPha (时间相关相位偏移)
            deltaPha1 = deltatime / (2.0 * np.pi)
            deltaPha1 = deltaPha1 - (int(deltaPha1 * 180 / np.pi) // 360) * 2 * np.pi
            deltaPha2 = deltaPha1 + np.pi / 4  # 45° offset between polarizations

            # 幅度
            examps1 = abs(f.ex1) * self.source_scale
            eyamps1 = abs(f.ey1) * self.source_scale
            hxamps1 = abs(f.hx1) * self.source_scale
            hyamps1 = abs(f.hy1) * self.source_scale
            hzamps1 = abs(f.hz1) * self.source_scale

            # 相位 - Ex/Ey 添加 +π 相位移
            exphss1 = np.angle(f.ex1) + np.pi
            eyphss1 = np.angle(f.ey1) + np.pi
            hxphss1 = np.angle(f.hx1)
            hyphss1 = np.angle(f.hy1)
            hzphss1 = np.angle(f.hz1)

            # 应用 deltaPha 相位偏移
            exphss1 = exphss1 + deltaPha1
            eyphss1 = eyphss1 + deltaPha1
            hxphss1 = hxphss1 + deltaPha1
            hyphss1 = hyphss1 + deltaPha1
            hzphss1 = hzphss1 + deltaPha1

            examps2 = abs(f.ex2) * self.source_scale
            eyamps2 = abs(f.ey2) * self.source_scale
            hxamps2 = abs(f.hx2) * self.source_scale
            hyamps2 = abs(f.hy2) * self.source_scale
            hzamps2 = abs(f.hz2) * self.source_scale

            exphss2 = np.angle(f.ex2) + np.pi
            eyphss2 = np.angle(f.ey2) + np.pi
            hxphss2 = np.angle(f.hx2)
            hyphss2 = np.angle(f.hy2)
            hzphss2 = np.angle(f.hz2)

            exphss2 = exphss2 + deltaPha2
            eyphss2 = eyphss2 + deltaPha2
            hxphss2 = hxphss2 + deltaPha2
            hyphss2 = hyphss2 + deltaPha2
            hzphss2 = hzphss2 + deltaPha2

            # 计算窗口长度 (约半个周期)
            wlen = max(int(self.sample_rate / 2 / freq) * 2, 2)

            # 生成单频分段信号并叠加到总输出
            freq_signal = self._generate_single_freq_segments(
                examps1,
                eyamps1,
                hxamps1,
                hyamps1,
                hzamps1,
                exphss1,
                eyphss1,
                hxphss1,
                hyphss1,
                hzphss1,
                examps2,
                eyamps2,
                hxamps2,
                hyamps2,
                hzamps2,
                exphss2,
                eyphss2,
                hxphss2,
                hyphss2,
                hzphss2,
                freq,
                n_samples,
                wlen,
                rng,
            )

            ex += freq_signal[0]
            ey += freq_signal[1]
            hx += freq_signal[2]
            hy += freq_signal[3]
            hz += freq_signal[4]

        return {
            "ex": ex,
            "ey": ey,
            "hx": hx,
            "hy": hy,
            "hz": hz,
            "sample_rate": self.sample_rate,
            "duration": duration,
            "n_samples": n_samples,
        }

    def _generate_single_freq_segments(
        self,
        examps1,
        eyamps1,
        hxamps1,
        hyamps1,
        hzamps1,
        exphss1,
        eyphss1,
        hxphss1,
        hyphss1,
        hzphss1,
        examps2,
        eyamps2,
        hxamps2,
        hyamps2,
        hzamps2,
        exphss2,
        eyphss2,
        hxphss2,
        hyphss2,
        hzphss2,
        freq,
        n_samples,
        wlen,
        rng,
    ):
        """
        生成单频分段时域信号

        流程:
        1. 生成若干段随机长度的时间序列
        2. 每段独立随机: 振幅、相位、极化角
        3. 边界应用余弦渐变窗保证连续性
        4. 拼接所有段
        """
        # 输出数组
        ex_out = np.zeros(n_samples)
        ey_out = np.zeros(n_samples)
        hx_out = np.zeros(n_samples)
        hy_out = np.zeros(n_samples)
        hz_out = np.zeros(n_samples)

        left = n_samples
        seg_start = 0

        while left > 0:
            # 随机段长度 - 高斯分布
            mean_len = int(self.sample_rate / freq * self.synthetic_periods)
            seg_len = int(abs(rng.normal(mean_len, mean_len / 2)))
            seg_len = max(seg_len, 10)  # 最小10样本
            seg_len = min(seg_len, left)  # 不超过剩余长度

            # 每段随机振幅
            da1 = rng.normal(1, 1)  # TM模式: 高斯N(1,1)
            dp1 = rng.random() * 2 * np.pi  # 随机相位
            da2 = rng.random() * 2  # TE模式: 均匀[0,2]
            dp2 = rng.random() * 2 * np.pi  # 随机相位

            # 每段随机极化角 - 控制TE/TM混合
            pol_angle = rng.random() * 2 * np.pi
            tm_weight = np.cos(pol_angle)
            te_weight = np.sin(pol_angle)

            # 构建幅度和相位数组
            amps1 = np.array(
                [
                    examps1 * da1,
                    eyamps1 * da1,
                    hxamps1 * da1,
                    hyamps1 * da1,
                    hzamps1 * da1,
                ]
            )
            phas1 = np.array(
                [
                    exphss1 + dp1,
                    eyphss1 + dp1,
                    hxphss1 + dp1,
                    hyphss1 + dp1,
                    hzphss1 + dp1,
                ]
            )

            amps2 = np.array(
                [
                    examps2 * da2,
                    eyamps2 * da2,
                    hxamps2 * da2,
                    hyamps2 * da2,
                    hzamps2 * da2,
                ]
            )
            phas2 = np.array(
                [
                    exphss2 + dp2,
                    eyphss2 + dp2,
                    hxphss2 + dp2,
                    hyphss2 + dp2,
                    hzphss2 + dp2,
                ]
            )

            # 生成时间段 (TM和TE分别生成再混合)
            seg_signal = self._generate_mixed_segment(
                amps1,
                phas1,
                amps2,
                phas2,
                freq,
                seg_len,
                wlen,
                tm_weight,
                te_weight,
                rng,
            )

            # 叠加到输出
            ex_out[seg_start : seg_start + seg_len] += seg_signal[0]
            ey_out[seg_start : seg_start + seg_len] += seg_signal[1]
            hx_out[seg_start : seg_start + seg_len] += seg_signal[2]
            hy_out[seg_start : seg_start + seg_len] += seg_signal[3]
            hz_out[seg_start : seg_start + seg_len] += seg_signal[4]

            seg_start += seg_len
            left -= seg_len

        return ex_out, ey_out, hx_out, hy_out, hz_out

    def _generate_mixed_segment(
        self, amps1, phas1, amps2, phas2, freq, seg_len, wlen, tm_weight, te_weight, rng
    ):
        """
        生成单个混合段

        每段内:
        1. 生成TM模式信号
        2. 生成TE模式信号
        3. 按极化角混合
        4. 边界应用余弦渐变窗保证与相邻段连续
        """
        t = np.arange(seg_len) / self.sample_rate

        # TM模式
        ex_tm = amps1[0] * np.cos(2 * np.pi * freq * t + phas1[0])
        ey_tm = amps1[1] * np.cos(2 * np.pi * freq * t + phas1[1])
        hx_tm = amps1[2] * np.cos(2 * np.pi * freq * t + phas1[2])
        hy_tm = amps1[3] * np.cos(2 * np.pi * freq * t + phas1[3])
        hz_tm = amps1[4] * np.cos(2 * np.pi * freq * t + phas1[4])

        # TE模式
        ex_te = amps2[0] * np.cos(2 * np.pi * freq * t + phas2[0])
        ey_te = amps2[1] * np.cos(2 * np.pi * freq * t + phas2[1])
        hx_te = amps2[2] * np.cos(2 * np.pi * freq * t + phas2[2])
        hy_te = amps2[3] * np.cos(2 * np.pi * freq * t + phas2[3])
        hz_te = amps2[4] * np.cos(2 * np.pi * freq * t + phas2[4])

        # 按极化角混合
        ex = ex_tm * tm_weight + ex_te * te_weight
        ey = ey_tm * tm_weight + ey_te * te_weight
        hx = hx_tm * tm_weight + hx_te * te_weight
        hy = hy_tm * tm_weight + hy_te * te_weight
        hz = hz_tm * tm_weight + hz_te * te_weight

        # 边界余弦窗 - 保证段间连续性
        if seg_len > wlen * 2:
            # 应用余弦渐变窗到边界
            window = self._cosine_window(seg_len, wlen)
            ex = ex * window
            ey = ey * window
            hx = hx * window
            hy = hy * window
            hz = hz * window

        return ex, ey, hx, hy, hz

    def _cosine_window(self, seg_len, wlen):
        """
        创建余弦渐变窗

        窗函数形状:
        - 开头wlen样本: 余弦上升 0→1
        - 中间: 恒定1
        - 结尾wlen样本: 余弦下降 1→0

        公式: window[i] = 0.5 * (1 - cos(π * i / wlen)) for i in [0, wlen)
              window[i] = 1.0 for i in [wlen, seg_len - wlen)
              window[i] = 0.5 * (1 + cos(π * (i - (seg_len - wlen)) / wlen)) for i in [seg_len - wlen, seg_len)
        """
        window = np.ones(seg_len)

        # 上升沿
        if wlen > 0 and wlen < seg_len // 2:
            for i in range(wlen):
                window[i] = 0.5 * (1 - np.cos(np.pi * i / wlen))
            # 下降沿
            for i in range(wlen):
                window[seg_len - 1 - i] = 0.5 * (1 - np.cos(np.pi * i / wlen))

        return window
