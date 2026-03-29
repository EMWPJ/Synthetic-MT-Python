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


class DeterministicTimeSeriesSynthesizer:
    """
    确定性MT时间序列合成器

    生成保持E-H相位关系的时间序列，使得处理后能精确恢复正演阻抗。

    原理:
    - Hy(t) = H_ref * cos(omega * t)
    - Ex(t) = Re{Zxy * Hy(t)} = |Zxy| * H_ref * cos(omega * t + arg(Zxy))

    这样 ExHy 互谱 = Zxy * |Hy|^2, HyHy = |Hy|^2
    因此 Zxy = ExHy / HyHy 精确成立
    """

    def __init__(self, sample_rate: float = 2400):
        self.sample_rate = sample_rate

    def generate_from_impedance(
        self,
        frequencies: np.ndarray,
        zxy: np.ndarray,
        zyx: np.ndarray,
        duration: float = 10.0,
        h_ref: float = 1e-3,  # 参考磁场振幅 (A/m)
    ) -> Dict:
        """
        从阻抗值生成确定性时间序列

        Args:
            frequencies: 频率数组 (Hz)
            zxy: 阻抗 Zxy (complex)
            zyx: 阻抗 Zyx (complex)
            duration: 时长 (秒)
            h_ref: 参考磁场振幅 (A/m)

        Returns:
            包含时间序列的字典
        """
        n_samples = int(self.sample_rate * duration)
        t = np.arange(n_samples) / self.sample_rate

        # 初始化输出
        ex = np.zeros(n_samples)
        ey = np.zeros(n_samples)
        hx = np.zeros(n_samples)
        hy = np.zeros(n_samples)
        hz = np.zeros(n_samples)

        # 对每个频率生成正弦波
        for i, f in enumerate(frequencies):
            omega = 2 * np.pi * f

            # Hy: 参考磁场 (余弦)
            hy_component = h_ref * np.cos(omega * t)
            hy += hy_component

            # Ex = Re{Zxy * Hy} = |Zxy| * H_ref * cos(omega*t + arg(Zxy))
            zxy_i = zxy[i]
            amplitude_ex = np.abs(zxy_i) * h_ref
            phase_ex = np.angle(zxy_i)  # zxy的相位
            ex += amplitude_ex * np.cos(omega * t + phase_ex)

            # Hx: 很小 (保持与正演一致)
            hx_small = h_ref * 0.01
            hx += hx_small * np.cos(omega * t)

            # Ey = Re{Zyx * Hx} ≈ 很小
            zyx_i = zyx[i]
            amplitude_ey = np.abs(zyx_i) * hx_small
            phase_ey = np.angle(zyx_i)
            ey += amplitude_ey * np.cos(omega * t + phase_ey)

        return {
            "ex": ex,
            "ey": ey,
            "hx": hx,
            "hy": hy,
            "hz": hz,
            "sample_rate": self.sample_rate,
            "duration": duration,
            "n_samples": n_samples,
            "frequencies": frequencies,
            "zxy": zxy,
            "zyx": zyx,
        }

    def generate_from_fields(
        self,
        fields: List,
        duration: float = 10.0,
    ) -> Dict:
        """
        从EMFields列表生成确定性时间序列

        Args:
            fields: EMFields列表 (来自MT1DForward.calculate_fields)
            duration: 时长 (秒)

        Returns:
            包含时间序列的字典
        """
        frequencies = np.array([f.freq for f in fields])
        zxy = np.array([f.zxy for f in fields])
        zyx = np.array([f.zyx for f in fields])

        # 使用TM模式 (mode 1) 的场分量
        # hx1很小, hy1是主力, ex1 = zxy * hy1
        h_ref = 1e-3  # A/m

        n_samples = int(self.sample_rate * duration)
        t = np.arange(n_samples) / self.sample_rate

        ex = np.zeros(n_samples)
        ey = np.zeros(n_samples)
        hx = np.zeros(n_samples)
        hy = np.zeros(n_samples)
        hz = np.zeros(n_samples)

        for i, f in enumerate(fields):
            omega = 2 * np.pi * f.freq

            # TM模式: Hy主力, Hx很小
            hy1 = np.abs(f.hy1) if f.hy1 != 0 else h_ref
            hx1 = np.abs(f.hx1) if f.hx1 != 0 else h_ref * 0.01

            # Hy(t) = |Hy1| * cos(omega*t)
            hy += hy1 * np.cos(omega * t)

            # Ex(t) = Re{Zxy * Hy(t)} = |Zxy| * |Hy1| * cos(omega*t + arg(Zxy))
            amplitude_ex = np.abs(f.zxy) * hy1
            phase_zxy = np.angle(f.zxy)
            ex += amplitude_ex * np.cos(omega * t + phase_zxy)

            # Hx(t)
            hx += hx1 * np.cos(omega * t)

            # Ey(t) = Re{Zyx * Hx(t)}
            amplitude_ey = np.abs(f.zyx) * hx1
            phase_zyx = np.angle(f.zyx)
            ey += amplitude_ey * np.cos(omega * t + phase_zyx)

        return {
            "ex": ex,
            "ey": ey,
            "hx": hx,
            "hy": hy,
            "hz": hz,
            "sample_rate": self.sample_rate,
            "duration": duration,
            "n_samples": n_samples,
            "frequencies": frequencies,
        }


# ============================================================================
# RANDOM_SEG_PARTIAL 时间序列合成器 (遵循Delphi算法)
# ============================================================================


class RandomSegmentTimeSeriesSynthesizer:
    """
    基于论文 Wang et al. (2023) 的 RANDOM_SEG_PARTIAL 算法
    完全遵循 Delphi SyntheticTimeSeries.pas 中的 RandomSegmentLengthPartiallyWindowedSyntheticSeries 实现

    该算法从正演结果生成MT时间序列，包含以下关键步骤：
    1. 两个正交极化 (TM和TE模式)
    2. 每段随机幅度 (高斯/均匀分布) 和随机相位 (均匀分布)
    3. 部分Hanning窗 - 仅在段边缘应用窗函数

    关键差异 (vs 旧版本):
    - 添加 +π 相位移到 Ex/Ey 相位 (遵循Delphi)
    - 添加 deltaPha 时间相关相位偏移 (45° 极化间差异)
    - 随机幅度/相位在每段内生成 (Delphi算法)

    参考: Delphi SyntheticTimeSeries.pas 中的 RandomSegmentLengthPartiallyWindowedSyntheticSeries
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

        # 初始化两个极化模式的累加数组
        ex1d1 = np.zeros(n_samples)
        ey1d1 = np.zeros(n_samples)
        hx1d1 = np.zeros(n_samples)
        hy1d1 = np.zeros(n_samples)
        hz1d1 = np.zeros(n_samples)

        ex1d2 = np.zeros(n_samples)
        ey1d2 = np.zeros(n_samples)
        hx1d2 = np.zeros(n_samples)
        hy1d2 = np.zeros(n_samples)
        hz1d2 = np.zeros(n_samples)

        # 时间偏移 (用于 deltaPha 计算)
        deltatime = start_time  # 秒

        for f in fields:
            freq = f.freq
            if freq <= 0:
                continue

            # 计算 deltaPha (时间相关相位偏移)
            # Delphi: deltaPha1 := deltatime / (2.0 * System.Pi);
            #         deltaPha1 := deltaPha1 - (Trunc(deltaPha1 * 180 / System.Pi) div 360) * 2 * System.Pi;
            #         deltaPha2 := deltaPha1 + System.Pi / 4;
            deltaPha1 = deltatime / (2.0 * np.pi)
            deltaPha1 = deltaPha1 - (int(deltaPha1 * 180 / np.pi) // 360) * 2 * np.pi
            deltaPha2 = deltaPha1 + np.pi / 4  # 45° offset between polarizations

            # 幅度 (遵循Delphi)
            # Delphi: examps1 := Ex1[I].Module * AllScale;
            examps1 = abs(f.ex1) * self.source_scale
            eyamps1 = abs(f.ey1) * self.source_scale
            hxamps1 = abs(f.hx1) * self.source_scale
            hyamps1 = abs(f.hy1) * self.source_scale
            hzamps1 = abs(f.hz1) * self.source_scale

            # 相位 - 遵循Delphi: Ex/Ey 添加 +π 相位移
            # Delphi: exphss1 := Ex1[I].Phase + System.Pi;
            exphss1 = np.angle(f.ex1) + np.pi
            eyphss1 = np.angle(f.ey1) + np.pi
            hxphss1 = np.angle(f.hx1)
            hyphss1 = np.angle(f.hy1)
            hzphss1 = np.angle(f.hz1)

            # 应用 deltaPha 相位偏移 (Delphi算法)
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
            # Delphi: wlen = max(int(sample_rate/2/freq) * 2, 2)
            wlen = max(int(self.sample_rate / 2 / freq) * 2, 2)

            # 极化1 (TM模式) - 随机段长度 + 部分窗
            # Delphi: da1 := RandG(1, 1); dp1 := Random * System.Pi * 2;
            # 注意: da1和dp1在每段循环内生成 (不是每极化生成一次)
            leftcount = n_samples
            while leftcount > 0:
                # Delphi: tmpcount := Ceil(Abs(SampleRate / freList[I] * (RandG(SyntheticPeriods, SyntheticPeriods / 2))));
                mean_len = int(self.sample_rate / freq * self.synthetic_periods)
                tmpcount = int(abs(rng.normal(mean_len, mean_len / 2)))
                while tmpcount < 10:
                    tmpcount = int(abs(rng.normal(mean_len, mean_len / 2)))

                # 确保段长度不会太小或太大
                if tmpcount > leftcount:
                    tmpcount = leftcount
                if leftcount - tmpcount < 10:
                    tmpcount = leftcount

                start = n_samples - leftcount

                # Delphi: 在循环内为每段生成新的随机值
                # TM模式: da1 = RandG(1,1) (高斯), dp1 = Random * 2π (均匀)
                da1 = rng.normal(1, 1)  # 高斯随机幅度 (mean=1, std=1)
                dp1 = rng.random() * 2 * np.pi  # 均匀随机相位 [0, 2π]

                # 生成时间段并应用部分窗
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
                self._generate_partial_windowed_segment(
                    amps1,
                    phas1,
                    freq,
                    tmpcount,
                    wlen,
                    start,
                    ex1d1,
                    ey1d1,
                    hx1d1,
                    hy1d1,
                    hz1d1,
                )

                leftcount -= tmpcount

            # 极化2 (TE模式) - 随机段长度 + 部分窗
            # Delphi: da2 := Random * 2; dp2 := Random * System.Pi * 2;
            # 注意: da2和dp2在每段循环内生成 (不是每极化生成一次)
            leftcount = n_samples
            while leftcount > 0:
                mean_len = int(self.sample_rate / freq * self.synthetic_periods)
                tmpcount = int(abs(rng.normal(mean_len, mean_len / 2)))
                while tmpcount < 10:
                    tmpcount = int(abs(rng.normal(mean_len, mean_len / 2)))

                if tmpcount > leftcount:
                    tmpcount = leftcount
                if leftcount - tmpcount < 10:
                    tmpcount = leftcount

                start = n_samples - leftcount

                # Delphi: 在循环内为每段生成新的随机值
                # TE模式: da2 = Random * 2 (均匀[0,2]), dp2 = Random * 2π (均匀)
                da2 = rng.random() * 2  # 均匀随机幅度 [0, 2]
                dp2 = rng.random() * 2 * np.pi  # 均匀随机相位 [0, 2π]

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
                self._generate_partial_windowed_segment(
                    amps2,
                    phas2,
                    freq,
                    tmpcount,
                    wlen,
                    start,
                    ex1d2,
                    ey1d2,
                    hx1d2,
                    hy1d2,
                    hz1d2,
                )

                leftcount -= tmpcount

        # 合并两个极化模式 - 使用极化角混合
        # 极化角控制TE/TM模式混合比例: ex = ex_TM * cos(θ) + ex_TE * sin(θ)
        # Delphi原始算法使用简单加法，这里增强为极化角混合模拟自然源极化方向变化
        # θ 在 [0, 2π) 范围内随机变化，每段使用相同的极化角实现一致的混合
        pol_angle = rng.random() * 2 * np.pi
        tm_weight = np.cos(pol_angle)
        te_weight = np.sin(pol_angle)

        ex = ex1d1 * tm_weight + ex1d2 * te_weight
        ey = ey1d1 * tm_weight + ey1d2 * te_weight
        hx = hx1d1 * tm_weight + hx1d2 * te_weight
        hy = hy1d1 * tm_weight + hy1d2 * te_weight
        hz = hz1d1 * tm_weight + hz1d2 * te_weight

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

    def _generate_partial_windowed_segment(
        self,
        amps: np.ndarray,
        phas: np.ndarray,
        freq: float,
        seg_len: int,
        wlen: int,
        start: int,
        ex_out: np.ndarray,
        ey_out: np.ndarray,
        hx_out: np.ndarray,
        hy_out: np.ndarray,
        hz_out: np.ndarray,
    ):
        """
        生成单个带部分窗的时间段

        Args:
            amps: [amp_ex, amp_ey, amp_hx, amp_hy, amp_hz] 幅度数组
            phas: [pha_ex, pha_ey, pha_hx, pha_hy, pha_hz] 相位数组
            freq: 频率
            seg_len: 段长度
            wlen: 窗长度
            start: 起始位置
            ex_out, ey_out, ...: 输出数组
        """
        # 创建时间序列
        t = np.arange(seg_len) / self.sample_rate

        # 各分量
        ex_seg = amps[0] * np.cos(2 * np.pi * freq * t + phas[0])
        ey_seg = amps[1] * np.cos(2 * np.pi * freq * t + phas[1])
        hx_seg = amps[2] * np.cos(2 * np.pi * freq * t + phas[2])
        hy_seg = amps[3] * np.cos(2 * np.pi * freq * t + phas[3])
        hz_seg = amps[4] * np.cos(2 * np.pi * freq * t + phas[4])

        # 部分窗 - Delphi算法
        # 如果段长度 > 2*wlen: 对首尾wlen样本应用窗
        # 否则: 不应用窗
        if seg_len > wlen * 2:
            end = start + seg_len

            # 应用窗到边缘
            ex_out[start : start + wlen] += ex_seg[:wlen] * np.hanning(wlen)
            ex_out[end - wlen : end] += ex_seg[-wlen:] * np.hanning(wlen)
            ex_out[start + wlen : end - wlen] += ex_seg[wlen:-wlen]

            ey_out[start : start + wlen] += ey_seg[:wlen] * np.hanning(wlen)
            ey_out[end - wlen : end] += ey_seg[-wlen:] * np.hanning(wlen)
            ey_out[start + wlen : end - wlen] += ey_seg[wlen:-wlen]

            hx_out[start : start + wlen] += hx_seg[:wlen] * np.hanning(wlen)
            hx_out[end - wlen : end] += hx_seg[-wlen:] * np.hanning(wlen)
            hx_out[start + wlen : end - wlen] += hx_seg[wlen:-wlen]

            hy_out[start : start + wlen] += hy_seg[:wlen] * np.hanning(wlen)
            hy_out[end - wlen : end] += hy_seg[-wlen:] * np.hanning(wlen)
            hy_out[start + wlen : end - wlen] += hy_seg[wlen:-wlen]

            hz_out[start : start + wlen] += hz_seg[:wlen] * np.hanning(wlen)
            hz_out[end - wlen : end] += hz_seg[-wlen:] * np.hanning(wlen)
            hz_out[start + wlen : end - wlen] += hz_seg[wlen:-wlen]
        else:
            # 短段不应用窗
            end = start + seg_len
            ex_out[start:end] += ex_seg
            ey_out[start:end] += ey_seg
            hx_out[start:end] += hx_seg
            hy_out[start:end] += hy_seg
            hz_out[start:end] += hz_seg
