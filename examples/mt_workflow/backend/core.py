"""
后台算法核心模块

整合所有MT算法模块，提供独立的计算逻辑。
此模块不依赖GUI，可独立运行和测试。
"""

import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING

# 仅用于类型检查的导入，避免循环依赖
if TYPE_CHECKING:
    from ..config import SegmentedAcquisitionConfig

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

            # 计算Hx-Hy相干性，用于判断是否使用简单比值法
            # coh_HxHy = |HxHy| / sqrt(HxHx * HyHy)
            # 当相干性接近1.0时，C_HH矩阵病态，直接求逆会失败
            hxhy_mag = np.abs(HxHy[i])
            hxhx_mag = np.abs(HxHx[i])
            hyhy_mag = np.abs(HyHy[i])
            coh_HxHy = hxhy_mag / np.sqrt(hxhx_mag * hyhy_mag + eps)

            # 使用正则化稳定矩阵求逆: C_HH_reg = C_HH + reg*I
            C_HH_reg = C_HH + reg * np.eye(2)
            det = C_HH_reg[0, 0] * C_HH_reg[1, 1] - C_HH_reg[0, 1] * C_HH_reg[1, 0]

            # 当Hx-Hy高度相关(相干性>0.95)时，使用简单比值法避免病态矩阵求逆
            # 这发生在TM/TE模式混合时，属于物理正常情况
            if coh_HxHy > 0.95 or np.abs(det) < eps:
                # 使用简单比值法: Zxy = ExHy/HyHy
                # 这是1D模型的正确方法，因为Ex和Hy直接通过Zxy关联
                # 而Hx与Ex无直接关系（除非Zxx≠0）
                Zxy[i] = ExHy[i] / (HyHy[i] + eps)
                Zyx[i] = EyHx[i] / (HxHx[i] + eps)
                # 1D约束: Zxx ≈ 0, Zyy ≈ 0, Zyx = -Zxy
                Zxx[i] = 0
                Zyy[i] = 0
                # 强制满足反对称约束
                Zyx[i] = -Zxy[i]
            else:
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
# 分段时间序列处理器
# ============================================================================


class SegmentedTimeSeriesProcessor:
    """
    分段时间序列处理器

    处理分段采集的数据 (HIGH/MED交替采集，带gap):
    1. 对每段数据进行FFT
    2. 计算交叉谱
    3. 跨段平均交叉谱
    4. 从平均交叉谱计算阻抗

    注意: 每个频段只能处理其频率范围内的数据
    - HIGH (TS3): 1-1000 Hz (周期 0.001-1.0s)
    - MED (TS4): 0.1-10 Hz (周期 0.1-10s)
    - LOW (TS5): 1e-6-1 Hz (周期 1-1e6s)
    """

    # Band to frequency range mapping
    BAND_FREQ_RANGES = {
        "HIGH": (1.0, 1000.0),  # Hz
        "MED": (0.1, 10.0),  # Hz
        "LOW": (1e-6, 1.0),  # Hz
    }

    def __init__(self, segments: List[Dict], sample_rate: float = None):
        """
        Args:
            segments: 分段数据列表，每段包含:
                - ex, ey, hx, hy, hz: 时间序列数组
                - band: 频段标识 (如 'HIGH', 'MED', 'LOW')
                - sample_rate: 采样率 (可选，会覆盖默认sample_rate)
            sample_rate: 默认采样率 (用于频率网格，当segments不含sample_rate时使用)
        """
        self.segments = segments
        self.sample_rate = sample_rate

        # 验证segments结构
        if not segments:
            raise ValueError("segments列表不能为空")

        # 按频段分组
        self._band_segments: Dict[str, List[Dict]] = {}
        for seg in segments:
            band = seg.get("band", "UNKNOWN")
            if band not in self._band_segments:
                self._band_segments[band] = []
            self._band_segments[band].append(seg)

        # 使用最长段的长度作为频率分辨率参考
        longest_seg = max(segments, key=lambda s: len(s.get("ex", [])))
        self._ref_n = len(longest_seg.get("ex", []))
        self._ref_sample_rate = longest_seg.get("sample_rate", sample_rate)

        # 预计算FFT频率数组
        self._freqs = np.fft.fftfreq(self._ref_n, 1.0 / self._ref_sample_rate)

    def estimate_impedance_at_periods(
        self, periods: np.ndarray, freq_tol: float = 0.1
    ) -> Dict:
        """
        估计指定周期处的阻抗

        对每个频段分别处理，然后合并结果

        Args:
            periods: 目标周期数组
            freq_tol: 频率容差 (默认0.1即10%)

        Returns:
            {
                "periods": np.ndarray,
                "Zxx", "Zxy", "Zyx", "Zyy": 阻抗张量
                "app_resistivity": 视电阻率
                "phase": 相位
                "coherence": 相干度
                "residual": 残差
            }
        """
        target_freqs = 1.0 / periods
        n = len(periods)

        # 初始化输出数组
        zxx_est = np.zeros(n, dtype=complex)
        zxy_est = np.zeros(n, dtype=complex)
        zyx_est = np.zeros(n, dtype=complex)
        zyy_est = np.zeros(n, dtype=complex)

        # 用于加权平均的权重
        total_weight = np.zeros(n)
        weighted_coherence = np.zeros(n)
        weighted_residual = np.zeros(n, dtype=complex)

        eps = 1e-20

        # 对每个频段分别处理
        for band, band_segs in self._band_segments.items():
            # 获取该频段的频率范围
            freq_range = self.BAND_FREQ_RANGES.get(band)
            if freq_range is None:
                continue
            band_freq_min, band_freq_max = freq_range

            # 计算该频段的平均交叉谱
            cross_spectra = self._compute_average_cross_spectra(band_segs)

            if cross_spectra is None:
                continue

            # 从交叉谱计算阻抗
            band_result = self._compute_impedance_from_cross_spectra(
                cross_spectra, target_freqs, freq_tol
            )

            if band_result is None:
                continue

            # 按段数加权平均，但只对目标频率在当前频段范围内的数据生效
            weight = len(band_segs)
            for i in range(n):
                if band_result["valid_mask"][i]:
                    # 检查目标频率是否在当前频段的有效范围内
                    target_freq = target_freqs[i]
                    if band_freq_min <= target_freq <= band_freq_max:
                        zxx_est[i] += band_result["Zxx"][i] * weight
                        zxy_est[i] += band_result["Zxy"][i] * weight
                        zyx_est[i] += band_result["Zyx"][i] * weight
                        zyy_est[i] += band_result["Zyy"][i] * weight
                        weighted_coherence[i] += band_result["coherence"][i] * weight
                        weighted_residual[i] += band_result["residual"][i] * weight
                        total_weight[i] += weight

        # 完成加权平均
        valid_mask = total_weight > 0
        for i in range(n):
            if valid_mask[i]:
                w = total_weight[i]
                zxx_est[i] /= w
                zxy_est[i] /= w
                zyx_est[i] /= w
                zyy_est[i] /= w
                weighted_coherence[i] /= w
                weighted_residual[i] /= w

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
            "coherence": weighted_coherence,
            "residual": weighted_residual,
        }

    def _find_signal_length(self, arr: np.ndarray) -> int:
        """
        查找数组中实际信号的长度（非零部分）

        由于分段采集数据在信号后会填充零值，需要找到实际信号长度
        来进行正确的FFT计算。

        Args:
            arr: 时间序列数组

        Returns:
            实际信号长度（不包括尾部零值）
        """
        # 找到最后一个非零值的索引
        nonzero_indices = np.nonzero(arr)[0]
        if len(nonzero_indices) == 0:
            return len(arr)
        return nonzero_indices[-1] + 1

    def _compute_average_cross_spectra(self, segments: List[Dict]) -> Optional[Dict]:
        """
        计算段列表的平均交叉谱

        对每个段只在其实际信号长度上进行FFT，避免对零填充部分进行FFT
        导致频谱失真。

        Args:
            segments: 同一频段的段列表

        Returns:
            包含平均交叉谱的字典，或None如果无效
        """
        if not segments:
            return None

        # 收集所有段的FFT结果
        all_ex_fft = []
        all_ey_fft = []
        all_hx_fft = []
        all_hy_fft = []
        all_seg_sample_rate = []
        all_n_sig = []

        for seg in segments:
            ex = seg.get("ex")
            ey = seg.get("ey")
            hx = seg.get("hx")
            hy = seg.get("hy")

            if ex is None or ey is None or hx is None or hy is None:
                continue

            seg_sample_rate = seg.get("sample_rate", self.sample_rate)

            if seg_sample_rate is None:
                continue

            # 找到实际信号长度（不包括尾部零填充）
            n_sig = min(
                self._find_signal_length(ex),
                self._find_signal_length(ey),
                self._find_signal_length(hx),
                self._find_signal_length(hy),
            )

            if n_sig < 10:  # 信号太短，跳过
                continue

            # 使用实际信号长度进行FFT
            ex_fft = np.fft.fft(ex[:n_sig] - np.mean(ex[:n_sig]))
            ey_fft = np.fft.fft(ey[:n_sig] - np.mean(ey[:n_sig]))
            hx_fft = np.fft.fft(hx[:n_sig] - np.mean(hx[:n_sig]))
            hy_fft = np.fft.fft(hy[:n_sig] - np.mean(hy[:n_sig]))

            all_ex_fft.append(ex_fft)
            all_ey_fft.append(ey_fft)
            all_hx_fft.append(hx_fft)
            all_hy_fft.append(hy_fft)
            all_seg_sample_rate.append(seg_sample_rate)
            all_n_sig.append(n_sig)

        if not all_ex_fft:
            return None

        # 使用参考段的频率数组（用于输出）
        ref_pos_mask = self._freqs > 0
        ref_freqs = self._freqs[ref_pos_mask]

        # 初始化累加器
        sum_ExHx = np.zeros(len(ref_freqs), dtype=complex)
        sum_ExHy = np.zeros(len(ref_freqs), dtype=complex)
        sum_EyHx = np.zeros(len(ref_freqs), dtype=complex)
        sum_EyHy = np.zeros(len(ref_freqs), dtype=complex)
        sum_HxHx = np.zeros(len(ref_freqs), dtype=complex)
        sum_HyHy = np.zeros(len(ref_freqs), dtype=complex)
        sum_HxHy = np.zeros(len(ref_freqs), dtype=complex)
        sum_HyHx = np.zeros(len(ref_freqs), dtype=complex)
        sum_ExEx = np.zeros(len(ref_freqs), dtype=complex)
        sum_EyEy = np.zeros(len(ref_freqs), dtype=complex)
        sum_ExEy = np.zeros(len(ref_freqs), dtype=complex)

        count = 0
        for idx in range(len(all_ex_fft)):
            ex_fft = all_ex_fft[idx]
            ey_fft = all_ey_fft[idx]
            hx_fft = all_hx_fft[idx]
            hy_fft = all_hy_fft[idx]
            seg_sr = all_seg_sample_rate[idx]
            n_sig = all_n_sig[idx]

            # 计算该段的频率数组（仅正频率部分）
            seg_freqs_full = np.fft.fftfreq(n_sig, 1.0 / seg_sr)
            seg_pos_mask = seg_freqs_full > 0
            seg_freqs = seg_freqs_full[seg_pos_mask]

            # 计算归一化因子（基于实际信号长度）
            norm = 2.0 / n_sig

            # 计算交叉谱
            ExHx = (ex_fft * np.conj(hx_fft))[seg_pos_mask] * norm
            ExHy = (ex_fft * np.conj(hy_fft))[seg_pos_mask] * norm
            EyHx = (ey_fft * np.conj(hx_fft))[seg_pos_mask] * norm
            EyHy = (ey_fft * np.conj(hy_fft))[seg_pos_mask] * norm
            HxHx = (hx_fft * np.conj(hx_fft))[seg_pos_mask] * norm
            HyHy = (hy_fft * np.conj(hy_fft))[seg_pos_mask] * norm
            HxHy = (hx_fft * np.conj(hy_fft))[seg_pos_mask] * norm
            HyHx = (hy_fft * np.conj(hx_fft))[seg_pos_mask] * norm
            ExEx = (ex_fft * np.conj(ex_fft))[seg_pos_mask] * norm
            EyEy = (ey_fft * np.conj(ey_fft))[seg_pos_mask] * norm
            ExEy = (ex_fft * np.conj(ey_fft))[seg_pos_mask] * norm

            # 插值到参考频率网格
            if len(seg_freqs) > 1:
                # 使用线性插值（对于频率轴是合理的近似）
                ExHx_interp = np.interp(ref_freqs, seg_freqs, np.abs(ExHx)) * np.exp(
                    1j * np.interp(ref_freqs, seg_freqs, np.angle(ExHx))
                )
                ExHy_interp = np.interp(ref_freqs, seg_freqs, np.abs(ExHy)) * np.exp(
                    1j * np.interp(ref_freqs, seg_freqs, np.angle(ExHy))
                )
                EyHx_interp = np.interp(ref_freqs, seg_freqs, np.abs(EyHx)) * np.exp(
                    1j * np.interp(ref_freqs, seg_freqs, np.angle(EyHx))
                )
                EyHy_interp = np.interp(ref_freqs, seg_freqs, np.abs(EyHy)) * np.exp(
                    1j * np.interp(ref_freqs, seg_freqs, np.angle(EyHy))
                )
                HxHx_interp = np.interp(ref_freqs, seg_freqs, np.abs(HxHx)) * np.exp(
                    1j * np.interp(ref_freqs, seg_freqs, np.angle(HxHx))
                )
                HyHy_interp = np.interp(ref_freqs, seg_freqs, np.abs(HyHy)) * np.exp(
                    1j * np.interp(ref_freqs, seg_freqs, np.angle(HyHy))
                )
                HxHy_interp = np.interp(ref_freqs, seg_freqs, np.abs(HxHy)) * np.exp(
                    1j * np.interp(ref_freqs, seg_freqs, np.angle(HxHy))
                )
                HyHx_interp = np.interp(ref_freqs, seg_freqs, np.abs(HyHx)) * np.exp(
                    1j * np.interp(ref_freqs, seg_freqs, np.angle(HyHx))
                )
                ExEx_interp = np.interp(ref_freqs, seg_freqs, np.abs(ExEx)) * np.exp(
                    1j * np.interp(ref_freqs, seg_freqs, np.angle(ExEx))
                )
                EyEy_interp = np.interp(ref_freqs, seg_freqs, np.abs(EyEy)) * np.exp(
                    1j * np.interp(ref_freqs, seg_freqs, np.angle(EyEy))
                )
                ExEy_interp = np.interp(ref_freqs, seg_freqs, np.abs(ExEy)) * np.exp(
                    1j * np.interp(ref_freqs, seg_freqs, np.angle(ExEy))
                )
            else:
                # 频率点数太少，无法插值
                continue

            # 累积交叉谱
            sum_ExHx += ExHx_interp
            sum_ExHy += ExHy_interp
            sum_EyHx += EyHx_interp
            sum_EyHy += EyHy_interp
            sum_HxHx += HxHx_interp
            sum_HyHy += HyHy_interp
            sum_HxHy += HxHy_interp
            sum_HyHx += HyHx_interp
            sum_ExEx += ExEx_interp
            sum_EyEy += EyEy_interp
            sum_ExEy += ExEy_interp

            count += 1

        if count == 0:
            return None

        # 平均
        return {
            "freqs": ref_freqs,
            "ExHx": sum_ExHx / count,
            "ExHy": sum_ExHy / count,
            "EyHx": sum_EyHx / count,
            "EyHy": sum_EyHy / count,
            "HxHx": sum_HxHx / count,
            "HyHy": sum_HyHy / count,
            "HxHy": sum_HxHy / count,
            "HyHx": sum_HyHx / count,
            "ExEx": sum_ExEx / count,
            "EyEy": sum_EyEy / count,
            "ExEy": sum_ExEy / count,
            "n_segs": count,
        }

    def _compute_impedance_from_cross_spectra(
        self,
        cross_spectra: Dict,
        target_freqs: np.ndarray,
        freq_tol: float = 0.1,
    ) -> Optional[Dict]:
        """
        从交叉谱计算阻抗张量

        Args:
            cross_spectra: 交叉谱字典
            target_freqs: 目标频率数组
            freq_tol: 频率容差

        Returns:
            包含阻抗和质量的字典，或None如果无效
        """
        freqs = cross_spectra.get("freqs")
        if freqs is None or len(freqs) == 0:
            return None

        n_freqs = len(freqs)
        n_targets = len(target_freqs)

        # 提取交叉谱
        ExHx = cross_spectra["ExHx"]
        ExHy = cross_spectra["ExHy"]
        EyHx = cross_spectra["EyHx"]
        EyHy = cross_spectra["EyHy"]
        HxHx = cross_spectra["HxHx"]
        HyHy = cross_spectra["HyHy"]
        HxHy = cross_spectra["HxHy"]
        HyHx = cross_spectra["HyHx"]
        ExEx = cross_spectra["ExEx"]
        EyEy = cross_spectra["EyEy"]
        ExEy = cross_spectra["ExEy"]

        # 初始化
        Zxx = np.zeros(n_freqs, dtype=complex)
        Zxy = np.zeros(n_freqs, dtype=complex)
        Zyx = np.zeros(n_freqs, dtype=complex)
        Zyy = np.zeros(n_freqs, dtype=complex)
        coherence = np.zeros(n_freqs, dtype=float)
        residual = np.zeros(n_freqs, dtype=complex)

        eps = 1e-20
        reg = 1e-10

        for i in range(n_freqs):
            # C_HH = [[HxHx, HxHy], [HyHx, HyHy]]
            # C_EH = [[ExHx, ExHy], [EyHx, EyHy]]
            C_HH = np.array([[HxHx[i], HxHy[i]], [HyHx[i], HyHy[i]]])
            C_EH = np.array([[ExHx[i], ExHy[i]], [EyHx[i], EyHy[i]]])

            # 计算Hx-Hy相干性
            hxhy_mag = np.abs(HxHy[i])
            hxhx_mag = np.abs(HxHx[i])
            hyhy_mag = np.abs(HyHy[i])
            coh_HxHy = hxhy_mag / np.sqrt(hxhx_mag * hyhy_mag + eps)

            # 正则化稳定矩阵求逆
            C_HH_reg = C_HH + reg * np.eye(2)
            det = C_HH_reg[0, 0] * C_HH_reg[1, 1] - C_HH_reg[0, 1] * C_HH_reg[1, 0]

            # 当Hx-Hy高度相关时，使用简单比值法
            if coh_HxHy > 0.95 or np.abs(det) < eps:
                Zxy[i] = ExHy[i] / (HyHy[i] + eps)
                Zyx[i] = EyHx[i] / (HxHx[i] + eps)
                Zxx[i] = 0
                Zyy[i] = 0
                Zyx[i] = -Zxy[i]
            else:
                # 标准最小二乘解: Z = C_EH * C_HH^(-1)
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

                # 计算相干性
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

        # 插值到目标频率
        zxx_target = np.zeros(n_targets, dtype=complex)
        zxy_target = np.zeros(n_targets, dtype=complex)
        zyx_target = np.zeros(n_targets, dtype=complex)
        zyy_target = np.zeros(n_targets, dtype=complex)
        coh_target = np.zeros(n_targets, dtype=float)
        res_target = np.zeros(n_targets, dtype=complex)
        valid_mask = np.zeros(n_targets, dtype=bool)

        for i, target_f in enumerate(target_freqs):
            idx = np.argmin(np.abs(freqs - target_f))
            if np.abs(freqs[idx] - target_f) / target_f < freq_tol:
                zxx_target[i] = Zxx[idx]
                zxy_target[i] = Zxy[idx]
                zyx_target[i] = Zyx[idx]
                zyy_target[i] = Zyy[idx]
                coh_target[i] = coherence[idx]
                res_target[i] = residual[idx]
                valid_mask[i] = True

        return {
            "Zxx": zxx_target,
            "Zxy": zxy_target,
            "Zyx": zyx_target,
            "Zyy": zyy_target,
            "coherence": coh_target,
            "residual": res_target,
            "valid_mask": valid_mask,
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
        synthetic_periods: float = 200.0,
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

        # 开始时间偏移 (用于计算源相位随时间的变化)
        start_time_offset = start_time  # 秒

        for f in fields:
            freq = f.freq
            if freq <= 0:
                continue

            # 计算源相位随时间的偏移 (时间相关相位偏移)
            # 公式: deltaPha = time / (2π)
            source1_phase_offset = start_time_offset / (2.0 * np.pi)
            source1_phase_offset = (
                source1_phase_offset
                - (int(source1_phase_offset * 180 / np.pi) // 360) * 2 * np.pi
            )
            source2_phase_offset = (
                source1_phase_offset + np.pi / 4
            )  # 45° offset between two orthogonal polarizations

            # TM模式 (Source 1) 幅度
            ex_tm_amp = abs(f.ex1) * self.source_scale
            ey_tm_amp = abs(f.ey1) * self.source_scale
            hx_tm_amp = abs(f.hx1) * self.source_scale
            hy_tm_amp = abs(f.hy1) * self.source_scale
            hz_tm_amp = abs(f.hz1) * self.source_scale

            # TM模式 (Source 1) 相位 - 直接使用fields中的相位
            # 注意: f.ex1 = zxy * hy, f.ey1 = zyx * hx 已经包含了正确的阻抗相位关系
            # 不需要额外添加 +pi
            ex_tm_phase = np.angle(f.ex1)
            ey_tm_phase = np.angle(f.ey1)
            hx_tm_phase = np.angle(f.hx1)
            hy_tm_phase = np.angle(f.hy1)
            hz_tm_phase = np.angle(f.hz1)

            # 应用源1相位偏移
            ex_tm_phase = ex_tm_phase + source1_phase_offset
            ey_tm_phase = ey_tm_phase + source1_phase_offset
            hx_tm_phase = hx_tm_phase + source1_phase_offset
            hy_tm_phase = hy_tm_phase + source1_phase_offset
            hz_tm_phase = hz_tm_phase + source1_phase_offset

            # TE模式 (Source 2) 幅度
            ex_te_amp = abs(f.ex2) * self.source_scale
            ey_te_amp = abs(f.ey2) * self.source_scale
            hx_te_amp = abs(f.hx2) * self.source_scale
            hy_te_amp = abs(f.hy2) * self.source_scale
            hz_te_amp = abs(f.hz2) * self.source_scale

            # TE模式 (Source 2) 相位 - 直接使用fields中的相位
            # 注意: f.ex2 = zxy * hy2, f.ey2 = zyx * hx2 已经包含了正确的阻抗相位关系
            # 不需要额外添加 +pi
            ex_te_phase = np.angle(f.ex2)
            ey_te_phase = np.angle(f.ey2)
            hx_te_phase = np.angle(f.hx2)
            hy_te_phase = np.angle(f.hy2)
            hz_te_phase = np.angle(f.hz2)

            # 应用源2相位偏移
            ex_te_phase = ex_te_phase + source2_phase_offset
            ey_te_phase = ey_te_phase + source2_phase_offset
            hx_te_phase = hx_te_phase + source2_phase_offset
            hy_te_phase = hy_te_phase + source2_phase_offset
            hz_te_phase = hz_te_phase + source2_phase_offset

            # 计算半周期窗口长度 (用于边界余弦窗)
            half_period_window = max(int(self.sample_rate / 2 / freq) * 2, 2)

            # 生成单频分段信号并叠加到总输出
            freq_signal = self._generate_single_freq_segments(
                # TM模式 (Source 1) 参数
                ex_tm_amp,
                ey_tm_amp,
                hx_tm_amp,
                hy_tm_amp,
                hz_tm_amp,
                ex_tm_phase,
                ey_tm_phase,
                hx_tm_phase,
                hy_tm_phase,
                hz_tm_phase,
                # TE模式 (Source 2) 参数
                ex_te_amp,
                ey_te_amp,
                hx_te_amp,
                hy_te_amp,
                hz_te_amp,
                ex_te_phase,
                ey_te_phase,
                hx_te_phase,
                hy_te_phase,
                hz_te_phase,
                # 频率和采样参数
                freq,
                n_samples,
                half_period_window,
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
        # TM模式 (Source 1) 幅度
        ex_tm_amp,
        ey_tm_amp,
        hx_tm_amp,
        hy_tm_amp,
        hz_tm_amp,
        # TM模式 (Source 1) 相位
        ex_tm_phase,
        ey_tm_phase,
        hx_tm_phase,
        hy_tm_phase,
        hz_tm_phase,
        # TE模式 (Source 2) 幅度
        ex_te_amp,
        ey_te_amp,
        hx_te_amp,
        hy_te_amp,
        hz_te_amp,
        # TE模式 (Source 2) 相位
        ex_te_phase,
        ey_te_phase,
        hx_te_phase,
        hy_te_phase,
        hz_te_phase,
        # 频率和采样参数
        freq,
        n_samples,
        half_period_window,
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

        samples_remaining = n_samples
        segment_position = 0

        while samples_remaining > 0:
            # 随机段长度 - 高斯分布
            mean_segment_length = int(self.sample_rate / freq * self.synthetic_periods)
            segment_length = int(
                abs(rng.normal(mean_segment_length, mean_segment_length / 2))
            )
            segment_length = max(segment_length, 10)  # 最小10样本
            segment_length = min(segment_length, samples_remaining)  # 不超过剩余长度

            # ========== 每段随机参数 ==========
            # TM模式: 高斯随机振幅因子 N(1,1), 均匀随机相位 [0, 2π)
            tm_amp_factor = rng.normal(1, 1)
            tm_phase_offset = rng.random() * 2 * np.pi

            # TE模式: 均匀随机振幅因子 [0, 2], 均匀随机相位 [0, 2π)
            te_amp_factor = rng.random() * 2
            te_phase_offset = rng.random() * 2 * np.pi

            # 每段随机极化角 - 控制TE/TM混合比例
            polarization_angle = rng.random() * 2 * np.pi
            tm_mode_weight = np.cos(polarization_angle)
            te_mode_weight = np.sin(polarization_angle)

            # 构建TM模式幅度和相位数组 (5通道: Ex,Ey,Hx,Hy,Hz)
            tm_amplitudes = np.array(
                [
                    ex_tm_amp * tm_amp_factor,
                    ey_tm_amp * tm_amp_factor,
                    hx_tm_amp * tm_amp_factor,
                    hy_tm_amp * tm_amp_factor,
                    hz_tm_amp * tm_amp_factor,
                ]
            )
            tm_phases = np.array(
                [
                    ex_tm_phase + tm_phase_offset,
                    ey_tm_phase + tm_phase_offset,
                    hx_tm_phase + tm_phase_offset,
                    hy_tm_phase + tm_phase_offset,
                    hz_tm_phase + tm_phase_offset,
                ]
            )

            # 构建TE模式幅度和相位数组
            te_amplitudes = np.array(
                [
                    ex_te_amp * te_amp_factor,
                    ey_te_amp * te_amp_factor,
                    hx_te_amp * te_amp_factor,
                    hy_te_amp * te_amp_factor,
                    hz_te_amp * te_amp_factor,
                ]
            )
            te_phases = np.array(
                [
                    ex_te_phase + te_phase_offset,
                    ey_te_phase + te_phase_offset,
                    hx_te_phase + te_phase_offset,
                    hy_te_phase + te_phase_offset,
                    hz_te_phase + te_phase_offset,
                ]
            )

            # 生成混合段 (TM和TE分别生成后按极化角混合)
            seg_signal = self._generate_mixed_segment(
                tm_amplitudes,
                tm_phases,
                te_amplitudes,
                te_phases,
                freq,
                segment_length,
                half_period_window,
                tm_mode_weight,
                te_mode_weight,
            )

            # 叠加到输出
            ex_out[segment_position : segment_position + segment_length] += seg_signal[
                0
            ]
            ey_out[segment_position : segment_position + segment_length] += seg_signal[
                1
            ]
            hx_out[segment_position : segment_position + segment_length] += seg_signal[
                2
            ]
            hy_out[segment_position : segment_position + segment_length] += seg_signal[
                3
            ]
            hz_out[segment_position : segment_position + segment_length] += seg_signal[
                4
            ]

            segment_position += segment_length
            samples_remaining -= segment_length

        return ex_out, ey_out, hx_out, hy_out, hz_out

    def _generate_mixed_segment(
        self,
        tm_amplitudes,
        tm_phases,
        te_amplitudes,
        te_phases,
        freq,
        segment_length,
        half_period_window,
        tm_mode_weight,
        te_mode_weight,
    ):
        """
        生成单个混合段

        每段内:
        1. 生成TM模式信号 (5通道)
        2. 生成TE模式信号 (5通道)
        3. 按极化角混合得到最终信号
        4. 边界应用余弦渐变窗保证与相邻段连续
        """
        t = np.arange(segment_length) / self.sample_rate

        # TM模式 (Source 1) - 5通道时域信号
        ex_tm = tm_amplitudes[0] * np.cos(2 * np.pi * freq * t + tm_phases[0])
        ey_tm = tm_amplitudes[1] * np.cos(2 * np.pi * freq * t + tm_phases[1])
        hx_tm = tm_amplitudes[2] * np.cos(2 * np.pi * freq * t + tm_phases[2])
        hy_tm = tm_amplitudes[3] * np.cos(2 * np.pi * freq * t + tm_phases[3])
        hz_tm = tm_amplitudes[4] * np.cos(2 * np.pi * freq * t + tm_phases[4])

        # TE模式 (Source 2) - 5通道时域信号
        ex_te = te_amplitudes[0] * np.cos(2 * np.pi * freq * t + te_phases[0])
        ey_te = te_amplitudes[1] * np.cos(2 * np.pi * freq * t + te_phases[1])
        hx_te = te_amplitudes[2] * np.cos(2 * np.pi * freq * t + te_phases[2])
        hy_te = te_amplitudes[3] * np.cos(2 * np.pi * freq * t + te_phases[3])
        hz_te = te_amplitudes[4] * np.cos(2 * np.pi * freq * t + te_phases[4])

        # 按极化角混合TM和TE模式
        # 物理意义: 自然源极化方向在空间上是变化的
        # 公式: E_final = E_TM * cos(θ) + E_TE * sin(θ)
        ex = ex_tm * tm_mode_weight + ex_te * te_mode_weight
        ey = ey_tm * tm_mode_weight + ey_te * te_mode_weight
        hx = hx_tm * tm_mode_weight + hx_te * te_mode_weight
        hy = hy_tm * tm_mode_weight + hy_te * te_mode_weight
        hz = hz_tm * tm_mode_weight + hz_te * te_mode_weight

        # 边界余弦窗 - 保证段间连续性
        # 只有当段长度大于2倍窗口长度时才应用窗函数
        if segment_length > half_period_window * 2:
            cosine_window = self._cosine_window(segment_length, half_period_window)
            ex = ex * cosine_window
            ey = ey * cosine_window
            hx = hx * cosine_window
            hy = hy * cosine_window
            hz = hz * cosine_window

        return ex, ey, hx, hy, hz

    def _cosine_window(self, segment_length, half_period_window):
        """
        创建余弦渐变窗 (Cosine Ramping Window)

        窗函数形状:
        - 开头 half_period_window 样本: 余弦上升 0→1 (渐入)
        - 中间: 恒定1 (保持)
        - 结尾 half_period_window 样本: 余弦下降 1→0 (渐出)

        物理意义: 使相邻段的边界处信号平滑过渡,避免突变

        公式:
          window[i] = 0.5 × (1 - cos(π × i / half_period_window))     当 i ∈ [0, half_period_window)  — 上升沿
          window[i] = 1.0                                              当 i ∈ [half_period_window, segment_length - half_period_window) — 平台
          window[i] = 0.5 × (1 + cos(π × (i - (segment_length - half_period_window)) / half_period_window))  当 i ∈ [segment_length - half_period_window, segment_length) — 下降沿
        """
        window = np.ones(segment_length)

        # 上升沿 (ramp-up) 和下降沿 (ramp-down)
        if half_period_window > 0 and half_period_window < segment_length // 2:
            # 上升沿: 0 → 1
            for i in range(half_period_window):
                window[i] = 0.5 * (1 - np.cos(np.pi * i / half_period_window))
            # 下降沿: 1 → 0
            for i in range(half_period_window):
                window[segment_length - 1 - i] = 0.5 * (
                    1 - np.cos(np.pi * i / half_period_window)
                )

        return window


# ============================================================================
# 分段交替采集时间序列合成器
# ============================================================================


class SegmentedTimeSeriesSynthesizer:
    """
    分段交替采集时间序列合成器

    实现多频段时间序列的交替采集模式:
    - HIGH (TS3, 2400Hz): 高频段采集
    - MED (TS4, 150Hz): 中频段采集
    - LOW (TS5, 15Hz): 低频段连续采集

    采集模式示例 (interval=300s, high_duration=2s, med_duration=16s):
        HIGH(2s) → gap(298s) → MED(16s) → gap(284s) → HIGH(2s) → ...

    Attributes:
        config: 分段采集配置
    """

    # Band to TSConfig name mapping
    BAND_TS_MAP = {
        "HIGH": "TS3",
        "MED": "TS4",
        "LOW": "TS5",
    }

    def __init__(self, acquisition_config: "SegmentedAcquisitionConfig"):
        """
        初始化分段交替采集合成器

        Args:
            acquisition_config: 分段采集配置
        """
        self.config = acquisition_config

    def generate(self, fields: List, seed: int = None) -> Dict:
        """
        生成分段交替采集的时间序列

        Args:
            fields: EMFields列表 (来自正演结果)
            seed: 随机种子

        Returns:
            {
                "segments": [
                    {
                        "band": "HIGH",
                        "start_time": float,  # 秒
                        "end_time": float,
                        "duration": float,
                        "sample_rate": float,
                        "n_samples": int,
                        "ex": np.ndarray, "ey": np.ndarray,
                        "hx": np.ndarray, "hy": np.ndarray, "hz": np.ndarray,
                    },
                    ...
                ],
                "low": {
                    "band": "LOW",
                    "start_time": float,
                    "end_time": float,
                    "duration": float,
                    "sample_rate": float,
                    "n_samples": int,
                    "ex": np.ndarray, "ey": np.ndarray,
                    "hx": np.ndarray, "hy": np.ndarray, "hz": np.ndarray,
                },
                "schedule": {...},  # acquisition_config参数副本
                "seed": int,
            }
        """
        # Generate acquisition schedule
        schedule = self.config.generate_schedule()
        total_duration = self.config.total_duration

        # Initialize result containers
        segments = []
        low_band_data = None

        # Process each segment in schedule
        for seg_idx, seg in enumerate(schedule):
            band = seg["band"]
            start_time = seg["start"]
            end_time = seg["end"]
            duration = seg["duration"]

            # Get TSConfig for this band
            ts_config_name = self.BAND_TS_MAP[band]
            ts_config = TS_CONFIGS[ts_config_name]

            # Filter fields to band's frequency range
            band_fields = [
                f for f in fields if ts_config.freq_min <= f.freq <= ts_config.freq_max
            ]

            # Skip if no fields in frequency range
            if not band_fields:
                # Create zero-filled segment
                n_samples = int(ts_config.sample_rate * duration)
                segments.append(
                    {
                        "band": band,
                        "start_time": start_time,
                        "end_time": end_time,
                        "duration": duration,
                        "sample_rate": ts_config.sample_rate,
                        "n_samples": n_samples,
                        "ex": np.zeros(n_samples),
                        "ey": np.zeros(n_samples),
                        "hx": np.zeros(n_samples),
                        "hy": np.zeros(n_samples),
                        "hz": np.zeros(n_samples),
                    }
                )
                continue

            # Create synthesizer for this band
            # Use default synthetic_periods=200 for good FFT resolution
            synthesizer = RandomSegmentTimeSeriesSynthesizer(
                sample_rate=ts_config.sample_rate,
                synthetic_periods=200.0,
                source_scale=1.0,
            )

            # Generate segment with seed for reproducibility
            segment_seed = (seed + seg_idx) if seed is not None else None
            ts_result = synthesizer.generate_from_fields(
                fields=band_fields,
                duration=duration,
                seed=segment_seed,
                start_time=start_time,
            )

            # Calculate gap padding if there's a gap after this segment
            next_seg = schedule[seg_idx + 1] if seg_idx + 1 < len(schedule) else None
            gap_samples = 0
            if next_seg:
                gap_duration = next_seg["start"] - end_time
                if gap_duration > 0:
                    gap_samples = int(ts_config.sample_rate * gap_duration)

            # Total samples = segment + gap
            n_segment_samples = ts_result["n_samples"]
            n_total_samples = n_segment_samples + gap_samples

            # Create full array with data + zero padding
            ex_full = np.zeros(n_total_samples)
            ey_full = np.zeros(n_total_samples)
            hx_full = np.zeros(n_total_samples)
            hy_full = np.zeros(n_total_samples)
            hz_full = np.zeros(n_total_samples)

            # Fill in the data
            ex_full[:n_segment_samples] = ts_result["ex"]
            ey_full[:n_segment_samples] = ts_result["ey"]
            hx_full[:n_segment_samples] = ts_result["hx"]
            hy_full[:n_segment_samples] = ts_result["hy"]
            hz_full[:n_segment_samples] = ts_result["hz"]

            segments.append(
                {
                    "band": band,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": duration,
                    "sample_rate": ts_config.sample_rate,
                    "n_samples": n_total_samples,
                    "ex": ex_full,
                    "ey": ey_full,
                    "hx": hx_full,
                    "hy": hy_full,
                    "hz": hz_full,
                }
            )

        # Generate continuous LOW band (TS5, 15Hz)
        low_config = TS_CONFIGS["TS5"]
        low_fields = [
            f for f in fields if low_config.freq_min <= f.freq <= low_config.freq_max
        ]

        if low_fields:
            low_synthesizer = RandomSegmentTimeSeriesSynthesizer(
                sample_rate=low_config.sample_rate,
                synthetic_periods=200.0,
                source_scale=1.0,
            )
            low_seed = (seed + len(schedule)) if seed is not None else None
            low_result = low_synthesizer.generate_from_fields(
                fields=low_fields,
                duration=total_duration,
                seed=low_seed,
                start_time=0.0,
            )
            low_band_data = {
                "band": "LOW",
                "start_time": 0.0,
                "end_time": total_duration,
                "duration": total_duration,
                "sample_rate": low_config.sample_rate,
                "n_samples": low_result["n_samples"],
                "ex": low_result["ex"],
                "ey": low_result["ey"],
                "hx": low_result["hx"],
                "hy": low_result["hy"],
                "hz": low_result["hz"],
            }
        else:
            # Create zero-filled low band
            n_low_samples = int(low_config.sample_rate * total_duration)
            low_band_data = {
                "band": "LOW",
                "start_time": 0.0,
                "end_time": total_duration,
                "duration": total_duration,
                "sample_rate": low_config.sample_rate,
                "n_samples": n_low_samples,
                "ex": np.zeros(n_low_samples),
                "ey": np.zeros(n_low_samples),
                "hx": np.zeros(n_low_samples),
                "hy": np.zeros(n_low_samples),
                "hz": np.zeros(n_low_samples),
            }

        return {
            "segments": segments,
            "low": low_band_data,
            "schedule": {
                "interval": self.config.interval,
                "high_duration": self.config.high_duration,
                "med_duration": self.config.med_duration,
                "total_duration": self.config.total_duration,
                "cycle_duration": self.config.cycle_duration,
            },
            "seed": seed,
        }
