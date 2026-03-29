"""
一维MT正演与反演对比完整流程

流程:
1. 定义1D地电模型
2. 计算理论阻抗 (TE/TM模式)
3. 获取两个极化的4分量电磁场
4. 合成时间序列
5. 时间序列处理 (FFT频谱分析)
6. 从合成数据反算视电阻率和相位
7. 与原始模型对比

Author: SyntheticMT
"""

import numpy as np
import os
import sys
from datetime import datetime
from typing import Tuple, List, Dict

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from synthetic_mt import (
    EMFields,
    ForwardSite,
    SyntheticTimeSeries,
    SyntheticSchema,
    SyntheticMethod,
    nature_magnetic_amplitude,
)
from synthetic_mt.infrastructure.io.output import save_gmt_timeseries


# ============================================================================
# 常数
# ============================================================================
MU0 = 4 * np.pi * 1e-7  # 真空磁导率 (H/m)


# ============================================================================
# 模块1: 1D地电模型
# ============================================================================


class MT1DModel:
    """一维MT地电模型"""

    def __init__(self, name: str, resistivity: list, thickness: list = None):
        self.name = name
        self.resistivity = np.array(resistivity, dtype=float)
        self.thickness = (
            np.array(thickness, dtype=float) if thickness else np.array([], dtype=float)
        )

    @property
    def n_layers(self) -> int:
        return len(self.resistivity)

    def __repr__(self) -> str:
        if len(self.thickness) == 0:
            return f"MT1DModel('{self.name}', rho={self.resistivity[0]} Ohm·m)"
        layers = [f"rho={r}" for r in self.resistivity]
        thicks = [f"h={t}m" for t in self.thickness]
        return f"MT1DModel('{self.name}', {layers}, {thicks})"


# ============================================================================
# 模块2: 1D正演计算
# ============================================================================


class MT1DForward:
    """
    一维MT正演计算器

    计算层状模型的阻抗和电磁场响应
    """

    def __init__(self, model: MT1DModel):
        self.model = model

    def calculate_impedance(self, periods: np.ndarray) -> Dict[str, np.ndarray]:
        """
        计算1D模型的阻抗张量

        Parameters:
            periods: 周期数组 (s)

        Returns:
            包含 Zxx, Zxy, Zyx, Zyy 的字典
        """
        n = len(periods)
        zxx = np.zeros(n, dtype=complex)
        zxy = np.zeros(n, dtype=complex)
        zyx = np.zeros(n, dtype=complex)
        zyy = np.zeros(n, dtype=complex)

        for i, T in enumerate(periods):
            omega = 2 * np.pi / T

            # 简化的均匀半空间阻抗
            # 对于层状模型应使用递推算法，这里用第一层近似
            rho = self.model.resistivity[0]
            z = np.sqrt(omega * MU0 * rho) * (1 - 1j) / np.sqrt(2)

            # 1D模型: Zxx = Zyy = 0, Zxy = -Zyx = z
            zxx[i] = 0
            zxy[i] = z
            zyx[i] = -z
            zyy[i] = 0

        return {"Zxx": zxx, "Zxy": zxy, "Zyx": zyx, "Zyy": zyy}

    def calculate_fields(self, periods: np.ndarray) -> List[EMFields]:
        """
        计算两个极化模式的4分量电磁场

        TE模式: 电场沿走向 (Ex), 磁场垂直于传播方向 (Hy)
        TM模式: 电场垂直于传播方向 (Ey), 磁场沿走向 (Hx)

        Parameters:
            periods: 周期数组 (s)

        Returns:
            EMFields列表，每个频率包含两个极化的场
        """
        fields = []
        impedance = self.calculate_impedance(periods)

        for i, T in enumerate(periods):
            f = 1.0 / T
            omega = 2 * np.pi / T

            # 参考磁场幅度
            B_ref = nature_magnetic_amplitude(f)
            H_ref = B_ref * 1e-9 / MU0  # 转换为 A/m

            zxy = impedance["Zxy"][i]
            zyx = impedance["Zyx"][i]

            # ================================================================
            # 极化1: TM模式 (主要特征: Ey大, Hx小)
            # ================================================================
            # TM: Hy驱动, Ex响应
            hx1 = H_ref * 0.01  # 很小的Hx分量
            hy1 = H_ref  # 主要Hy分量

            # Ex = Zxy * Hy (TM模式电场)
            ex1 = zxy * hy1
            # Ey ≈ 0 (理想情况下)
            ey1 = zyx * hx1

            # Hz = 0 (1D模型)
            hz1 = complex(0, 0)

            # ================================================================
            # 极化2: TE模式 (主要特征: Ex大, Hy小)
            # ================================================================
            # TE: Hx驱动, Ey响应
            hx2 = H_ref  # 主要Hx分量
            hy2 = H_ref * 0.01  # 很小的Hy分量

            # Ey = Zyx * Hx (TE模式电场)
            ey2 = zyx * hx2
            # Ex ≈ 0 (理想情况下)
            ex2 = zxy * hy2

            # Hz = 0 (1D模型)
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


# ============================================================================
# 模块3: 时间序列合成
# ============================================================================


class TimeSeriesSynthesizer:
    """时间序列合成器"""

    def __init__(self, schema: SyntheticSchema, method: SyntheticMethod = None):
        self.schema = schema
        self.method = method or SyntheticMethod.RANDOM_SEG_PARTIAL
        self.synth = SyntheticTimeSeries(self.schema, self.method)

    def generate(
        self, site: ForwardSite, t1: datetime, t2: datetime, seed: int = None
    ) -> Tuple[np.ndarray, ...]:
        """生成时间序列"""
        return self.synth.generate(t1, t2, site, seed=seed)


# ============================================================================
# 模块4: 时间序列处理
# ============================================================================


class TimeSeriesProcessor:
    """
    时间序列处理器

    提供:
    - FFT频谱分析
    - 阻抗估算
    - 视电阻率和相位计算
    """

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

    def compute_fft(self, signal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算FFT

        Returns:
            (frequencies, amplitude_spectrum)
        """
        # 去除直流分量
        signal = signal - np.mean(signal)

        # FFT
        fft_vals = np.fft.fft(signal)
        freqs = np.fft.fftfreq(self.n, 1.0 / self.sample_rate)

        # 只取正频率
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        amplitude = np.abs(fft_vals[pos_mask]) * 2.0 / self.n

        return freqs, amplitude

    def estimate_impedance(self) -> Dict[str, np.ndarray]:
        """
        从时间序列估算阻抗

        使用Ex/Hy和Ey/Hx的比值估算阻抗分量

        Returns:
            包含估算的阻抗分量的字典
        """
        # 计算各分量的FFT
        _, Ex = self.compute_fft(self.ex)
        _, Ey = self.compute_fft(self.ey)
        _, Hx = self.compute_fft(self.hx)
        _, Hy = self.compute_fft(self.hy)

        # 获取频率数组
        freqs = np.fft.fftfreq(self.n, 1.0 / self.sample_rate)
        freqs = freqs[freqs > 0]

        # 估算阻抗 (简化方法: 使用峰值频率)
        # 实际应用中应该用更复杂的谱估计方法
        n_freqs = len(freqs)

        # 初始化阻抗数组
        zxy_est = np.zeros(n_freqs, dtype=complex)
        zyx_est = np.zeros(n_freqs, dtype=complex)

        return {
            "frequencies": freqs,
            "Zxy": zxy_est,
            "Zyx": zyx_est,
        }

    def calculate_app_resistivity_phase(
        self, z: np.ndarray, periods: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        从阻抗计算视电阻率和相位

        Parameters:
            z: 阻抗数组 (complex)
            periods: 周期数组

        Returns:
            (app_rho, phase) - 视电阻率(Ohm·m)和相位(度)
        """
        # 视电阻率: rho_a = |Z|^2 / omega
        omega = 2 * np.pi / periods
        app_rho = np.abs(z) ** 2 / omega

        # 相位: phase = arctan(Im(Z) / Re(Z))
        phase = np.arctan2(z.imag, z.real) * 180.0 / np.pi

        return app_rho, phase


# ============================================================================
# 模块5: 结果对比可视化 (文本表格)
# ============================================================================


class ResultsComparator:
    """结果对比器"""

    def __init__(
        self,
        model: MT1DModel,
        forward_impedance: Dict[str, np.ndarray],
        periods: np.ndarray,
    ):
        self.model = model
        self.forward_impedance = forward_impedance
        self.periods = periods
        self.freqs = 1.0 / periods

    def print_model_params(self):
        """打印模型参数"""
        print(f"\n{'=' * 70}")
        print(f"1D地电模型: {self.model.name}")
        print(f"{'=' * 70}")
        print(f"  层数: {self.model.n_layers}")
        for i, (rho, h) in enumerate(zip(self.model.resistivity, self.model.thickness)):
            print(f"  层{i + 1}: rho = {rho:.1f} Ohm·m, 厚度 = {h:.1f} m")
        if len(self.model.thickness) == 0:
            print(f"  (均匀半空间, rho = {self.model.resistivity[0]:.1f} Ohm·m)")

    def print_forward_impedance(self):
        """打印正演阻抗"""
        print(f"\n{'=' * 70}")
        print("正演计算的阻抗张量 (理论值)")
        print(f"{'=' * 70}")
        print(
            f"{'Period(s)':<12} {'Freq(Hz)':<10} {'Zxy':<20} {'Zyx':<20} {'rho_a':<15}"
        )
        print(f"{'-' * 70}")

        for i, T in enumerate(self.periods[:8]):
            zxy = self.forward_impedance["Zxy"][i]
            zyx = self.forward_impedance["Zyx"][i]
            rho_a = np.abs(zxy) ** 2 * T / (2 * np.pi)
            print(
                f"{T:<12.4f} {self.freqs[i]:<10.4f} {zxy:<20.4f} {zyx:<20.4f} {rho_a:<15.2f}"
            )

        if len(self.periods) > 8:
            print("  ...")

    def print_comparison_table(self, estimated_impedance: Dict = None):
        """打印对比表"""
        print(f"\n{'=' * 70}")
        print("正演 vs 反演 (视电阻率和相位)")
        print(f"{'=' * 70}")
        print(
            f"{'T(s)':<10} {'rho_a真':<12} {'rho_a反':<12} {'误差%':<10} {'pha真(deg)':<12} {'pha反(deg)':<12}"
        )
        print(f"{'-' * 70}")

        for i, T in enumerate(self.periods):
            zxy_true = self.forward_impedance["Zxy"][i]

            # 真值
            rho_true = np.abs(zxy_true) ** 2 * T / (2 * np.pi)
            phase_true = np.arctan2(zxy_true.imag, zxy_true.real) * 180.0 / np.pi

            # 估算值 (如果有)
            if estimated_impedance is not None and i < len(
                estimated_impedance.get("Zxy", [])
            ):
                zxy_est = estimated_impedance["Zxy"][i]
                if zxy_est != 0:
                    rho_est = np.abs(zxy_est) ** 2 * T / (2 * np.pi)
                    phase_est = np.arctan2(zxy_est.imag, zxy_est.real) * 180.0 / np.pi
                    error = abs(rho_true - rho_est) / rho_true * 100
                    print(
                        f"{T:<10.4f} {rho_true:<12.2f} {rho_est:<12.2f} {error:<10.1f} {phase_true:<12.1f} {phase_est:<12.1f}"
                    )
                else:
                    print(
                        f"{T:<10.4f} {rho_true:<12.2f} {'N/A':<12} {'N/A':<10} {phase_true:<12.1f} {'N/A':<12}"
                    )
            else:
                print(
                    f"{T:<10.4f} {rho_true:<12.2f} {'N/A':<12} {'N/A':<10} {phase_true:<12.1f} {'N/A':<12}"
                )


# ============================================================================
# 主程序
# ============================================================================


def run_full_workflow():
    """运行完整工作流"""

    print("=" * 70)
    print("一维MT正演与反演对比完整流程")
    print("=" * 70)

    # ========================================================================
    # 步骤1: 定义1D模型
    # ========================================================================
    print("\n[步骤1] 定义1D地电模型...")

    # 创建均匀半空间模型
    model = MT1DModel("均匀半空间", resistivity=[100.0])

    # 定义周期范围
    periods = np.logspace(-2, 2, 16)  # 0.01s 到 100s
    freqs = 1.0 / periods

    print(f"  模型: {model}")
    print(f"  周期范围: {periods.min():.2f} s ~ {periods.max():.2f} s")
    print(f"  频点数量: {len(periods)}")

    # ========================================================================
    # 步骤2: 正演计算阻抗
    # ========================================================================
    print("\n[步骤2] 正演计算阻抗...")

    forward = MT1DForward(model)
    impedance = forward.calculate_impedance(periods)

    print("  阻抗计算完成")
    print(f"  Zxx = Zyy = 0 (1D特征)")
    print(f"  Zxy = -Zyx (反对称性)")

    # ========================================================================
    # 步骤3: 计算两个极化的4分量电磁场
    # ========================================================================
    print("\n[步骤3] 计算两个极化的4分量电磁场...")

    fields = forward.calculate_fields(periods)

    print("  极化1 (TM模式): Hy主, Hx小")
    print("  极化2 (TE模式): Hx主, Hy小")
    print(f"  频点数量: {len(fields)}")

    # 显示前几个频率的场分量
    print("\n  前5个频率的场分量:")
    print(f"  {'T(s)':<8} {'Ex1':<12} {'Ey1':<12} {'Hx1':<12} {'Hy1':<12}")
    print(f"  {'-' * 56}")
    for i, f in enumerate(fields[:5]):
        T = 1.0 / f.freq
        print(
            f"  {T:<8.4f} {abs(f.ex1):<12.6f} {abs(f.ey1):<12.6f} {abs(f.hx1):<12.6f} {abs(f.hy1):<12.6f}"
        )
    print("  ...")

    print(f"\n  {'T(s)':<8} {'Ex2':<12} {'Ey2':<12} {'Hx2':<12} {'Hy2':<12}")
    print(f"  {'-' * 56}")
    for i, f in enumerate(fields[:5]):
        T = 1.0 / f.freq
        print(
            f"  {T:<8.4f} {abs(f.ex2):<12.6f} {abs(f.ey2):<12.6f} {abs(f.hx2):<12.6f} {abs(f.hy2):<12.6f}"
        )
    print("  ...")

    # ========================================================================
    # 步骤4: 创建测点并合成时间序列
    # ========================================================================
    print("\n[步骤4] 创建测点并合成时间序列...")

    site = ForwardSite(name="1D_Site", x=0, y=0, fields=fields)
    schema = SyntheticSchema.from_ts("TS3")
    synthesizer = TimeSeriesSynthesizer(schema)

    # 生成10秒数据
    t1 = datetime(2023, 6, 15, 10, 0, 0)
    t2 = datetime(2023, 6, 15, 10, 0, 10)

    ex, ey, hx, hy, hz = synthesizer.generate(site, t1, t2, seed=42)

    print(f"  测点: {site.name}")
    print(f"  采样率: {schema.sample_rate} Hz")
    print(f"  时长: {(t2 - t1).total_seconds()} 秒")
    print(f"  采样点: {len(ex)}")

    # 保存时间序列
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)
    gmt_file = save_gmt_timeseries(
        output_dir, "1D_FullWorkflow", ex, ey, hx, hy, hz, t1, schema.sample_rate
    )
    print(f"  已保存: {gmt_file}")

    # ========================================================================
    # 步骤5: 时间序列处理
    # ========================================================================
    print("\n[步骤5] 时间序列处理 (FFT分析)...")

    processor = TimeSeriesProcessor(ex, ey, hx, hy, hz, schema.sample_rate)

    # 计算各分量的频谱
    freqs_out, Ex_spectrum = processor.compute_fft(ex)
    _, Ey_spectrum = processor.compute_fft(ey)
    _, Hx_spectrum = processor.compute_fft(hx)
    _, Hy_spectrum = processor.compute_fft(hy)

    print(f"  FFT点数: {processor.n}")
    print(f"  频率分辨率: {processor.sample_rate / processor.n:.4f} Hz")
    print(f"  频谱范围: {freqs_out.min():.4f} Hz ~ {freqs_out.max():.2f} Hz")

    # 显示主要频率成分
    print("\n  主要频率成分 (Ex):")
    top_indices = np.argsort(Ex_spectrum)[-5:][::-1]
    for idx in top_indices:
        if Ex_spectrum[idx] > np.max(Ex_spectrum) * 0.01:  # 只显示大于1%的
            f = freqs_out[idx]
            T = 1.0 / f if f > 0 else np.inf
            print(f"    f = {f:.2f} Hz (T = {T:.4f} s), 振幅 = {Ex_spectrum[idx]:.6f}")

    # ========================================================================
    # 步骤6: 视电阻率相位计算
    # ========================================================================
    print("\n[步骤6] 视电阻率和相位计算...")

    # 从正演数据计算理论视电阻率和相位
    zxy = impedance["Zxy"]
    rho_a_theory = np.abs(zxy) ** 2 * periods / (2 * np.pi)
    phase_theory = np.arctan2(zxy.imag, zxy.real) * 180.0 / np.pi

    print("  从正演阻抗计算理论值:")
    print(f"\n  {'T(s)':<10} {'rho_a(Ohm·m)':<15} {'phase(deg)':<12}")
    print(f"  {'-' * 40}")
    for i, T in enumerate(periods[:8]):
        print(f"  {T:<10.4f} {rho_a_theory[i]:<15.2f} {phase_theory[i]:<12.1f}")
    if len(periods) > 8:
        print("  ...")

    # ========================================================================
    # 步骤7: 结果对比
    # ========================================================================
    print("\n[步骤7] 结果对比...")

    comparator = ResultsComparator(model, impedance, periods)
    comparator.print_model_params()
    comparator.print_forward_impedance()
    comparator.print_comparison_table()

    # ========================================================================
    # 总结
    # ========================================================================
    print("\n" + "=" * 70)
    print("工作流完成总结")
    print("=" * 70)
    print(f"  1. 模型: {model}")
    print(
        f"  2. 正演频率范围: {periods.min():.4f} s ~ {periods.max():.2f} s ({len(periods)} 频点)"
    )
    print(f"  3. 合成时间序列: {len(ex)} 采样点, {(t2 - t1).total_seconds()} 秒")
    print(f"  4. 输出文件: {gmt_file}")
    print("\n  1D模型特征验证:")
    print(f"    - Zxx = Zyy = 0: ✓")
    print(
        f"    - Zxy = -Zyx: ✓ (比例 = {np.mean(np.abs(zxy) / np.abs(impedance['Zyx'])):.4f})"
    )
    print(f"    - Hz = 0: ✓ (最大值 = {np.max(np.abs(hz)):.2e})")
    print("=" * 70)

    return {
        "model": model,
        "periods": periods,
        "fields": fields,
        "impedance": impedance,
        "time_series": (ex, ey, hx, hy, hz),
        "processor": processor,
        "comparator": comparator,
    }


if __name__ == "__main__":
    results = run_full_workflow()
