"""
一维MT模型正演合成时间序列示例 - 模块化版本

演示如何使用SyntheticMT基于1D地电模型正演结果合成MT时间序列。

模块化设计:
- model_1d: 1D地电模型定义与阻抗计算
- data_generator: 时间序列生成器
- validators: 1D特征验证器

Author: SyntheticMT
"""

import numpy as np
import os
import sys

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
    save_gmt_timeseries,
    save_csv_timeseries,
)
from datetime import datetime


# ============================================================================
# 模块1: 1D地电模型定义
# ============================================================================


class MT1DModel:
    """
    一维MT地电模型

    Attributes:
        name: 模型名称
        resistivity: 各层电阻率 (Ohm·m)
        thickness: 各层厚度 (m)，最后一层为半空间
    """

    def __init__(self, name: str, resistivity: list, thickness: list = None):
        self.name = name
        self.resistivity = np.array(resistivity)
        self.thickness = np.array(thickness) if thickness else np.array([])

    @property
    def n_layers(self) -> int:
        """层数"""
        return len(self.resistivity)

    @property
    def is_halfspace(self) -> bool:
        """是否为半空间模型"""
        return len(self.thickness) == 0 or len(self.thickness) == self.n_layers - 1

    def __repr__(self) -> str:
        if self.is_halfspace:
            return (
                f"MT1DModel('{self.name}', rho={self.resistivity[0]} Ohm·m, halfspace)"
            )
        return f"MT1DModel('{self.name}', rho={self.resistivity}, h={self.thickness})"


def calculate_1d_impedance(model: MT1DModel, periods: np.ndarray) -> tuple:
    """
    计算一维MT模型的阻抗响应

    Parameters:
        model: 1D地电模型
        periods: 周期数组 (s)

    Returns:
        (zxy, zyx): 阻抗数组
    """
    mu0 = 4 * np.pi * 1e-7
    n = len(periods)
    zxy = np.zeros(n, dtype=complex)
    zyx = np.zeros(n, dtype=complex)

    rho = model.resistivity[0]  # 使用第一层

    for i, T in enumerate(periods):
        omega = 2 * np.pi / T
        z = np.sqrt(omega * mu0 * rho) * (1 - 1j) / np.sqrt(2)
        zxy[i] = z
        zyx[i] = -z

    return zxy, zyx


# ============================================================================
# 模块2: EMFields构建器
# ============================================================================


class EMFieldsBuilder:
    """EMFields构建器"""

    def __init__(self, model: MT1DModel):
        self.model = model
        self.mu0 = 4 * np.pi * 1e-7

    def build_fields(self, periods: np.ndarray) -> list:
        """
        从1D模型构建EMFields列表

        Parameters:
            periods: 周期数组 (s)

        Returns:
            EMFields列表
        """
        zxy, zyx = calculate_1d_impedance(self.model, periods)
        fields = []

        for i, T in enumerate(periods):
            f = 1.0 / T
            B_ref = nature_magnetic_amplitude(f)
            H_ref = B_ref * 1e-9 / self.mu0

            # 极化1: TM模式 (Hy为主)
            hx1, hy1 = H_ref * 0.01, H_ref
            ex1 = zxy[i] * hy1
            ey1 = zyx[i] * hx1

            # 极化2: TE模式 (Hx为主)
            hx2, hy2 = H_ref, H_ref * 0.01
            ex2 = zxy[i] * hy2
            ey2 = zyx[i] * hx2

            fields.append(
                EMFields(
                    freq=f,
                    ex1=complex(ex1.real, ex1.imag),
                    ey1=complex(ey1.real, ey1.imag),
                    hx1=complex(hx1, 0),
                    hy1=complex(hy1, 0),
                    hz1=complex(0, 0),
                    ex2=complex(ex2.real, ex2.imag),
                    ey2=complex(ey2.real, ey2.imag),
                    hx2=complex(hx2, 0),
                    hy2=complex(hy2, 0),
                    hz2=complex(0, 0),
                    zxx=complex(0, 0),
                    zxy=zxy[i],
                    zyx=zyx[i],
                    zyy=complex(0, 0),
                    tzx=complex(0, 0),
                    tzy=complex(0, 0),
                )
            )

        return fields


# ============================================================================
# 模块3: 时间序列生成器
# ============================================================================


class TimeSeriesGenerator:
    """MT时间序列生成器"""

    def __init__(self, schema: SyntheticSchema, method: SyntheticMethod = None):
        self.schema = schema
        self.method = method or SyntheticMethod.RANDOM_SEG_PARTIAL
        self.synth = SyntheticTimeSeries(self.schema, self.method)

    def generate(
        self, site: ForwardSite, t1: datetime, t2: datetime, seed: int = None
    ) -> tuple:
        """生成时间序列"""
        return self.synth.generate(t1, t2, site, seed=seed)

    @staticmethod
    def create_site(name: str, x: float, y: float, fields: list) -> ForwardSite:
        """创建测点"""
        return ForwardSite(name=name, x=x, y=y, fields=fields)


# ============================================================================
# 模块4: 1D特征验证器
# ============================================================================


class Model1DValidator:
    """1D模型特征验证器"""

    def __init__(self, fields: list, time_series: tuple = None):
        self.fields = fields
        self.ex, self.ey, self.hx, self.hy, self.hz = (
            time_series if time_series else (None,) * 5
        )

    def check_impedance_symmetry(self) -> dict:
        """检查阻抗反对称性: Zxy = -Zyx"""
        zxy_mag = np.mean([abs(f.zxy) for f in self.fields])
        zyx_mag = np.mean([abs(f.zyx) for f in self.fields])
        ratio = zxy_mag / zyx_mag if zyx_mag > 0 else 0

        return {
            "passed": 0.95 < ratio < 1.05,
            "ratio": ratio,
            "expected": 1.0,
            "message": f"Zxy/Zyx = {ratio:.4f}",
        }

    def check_zero_diagonal(self) -> dict:
        """检查阻抗对角元为零: Zxx = Zyy = 0"""
        zxx_max = max([abs(f.zxx) for f in self.fields]) if self.fields else 0
        zyy_max = max([abs(f.zyy) for f in self.fields]) if self.fields else 0

        return {
            "passed": zxx_max < 1e-10 and zyy_max < 1e-10,
            "zxx_max": zxx_max,
            "zyy_max": zyy_max,
            "message": f"Zxx_max={zxx_max:.2e}, Zyy_max={zyy_max:.2e}",
        }

    def check_zero_tipper(self) -> dict:
        """检查Tipper为零"""
        tzx_max = max([abs(f.tzx) for f in self.fields]) if self.fields else 0
        tzy_max = max([abs(f.tzy) for f in self.fields]) if self.fields else 0

        return {
            "passed": tzx_max < 1e-10 and tzy_max < 1e-10,
            "tzx_max": tzx_max,
            "tzy_max": tzy_max,
            "message": f"Tzx_max={tzx_max:.2e}, Tzy_max={tzy_max:.2e}",
        }

    def check_zero_vertical(self) -> dict:
        """检查垂直分量Hz ≈ 0"""
        if self.hz is None:
            return {"passed": False, "message": "No time series provided"}

        hz_max = np.max(np.abs(self.hz))
        hxy_max = max(np.max(np.abs(self.hx)), np.max(np.abs(self.hy)))
        ratio = hz_max / hxy_max if hxy_max > 0 else 0

        return {
            "passed": ratio < 1e-6,
            "ratio": ratio,
            "message": f"Hz_max/Hxy_max = {ratio:.2e}",
        }

    def validate_all(self) -> dict:
        """执行所有验证"""
        results = {
            "impedance_symmetry": self.check_impedance_symmetry(),
            "zero_diagonal": self.check_zero_diagonal(),
            "zero_tipper": self.check_zero_tipper(),
            "zero_vertical": self.check_zero_vertical(),
        }

        results["all_passed"] = all(r["passed"] for r in results.values())
        return results


# ============================================================================
# 模块5: 统计信息
# ============================================================================


class TimeSeriesStats:
    """时间序列统计"""

    def __init__(self, ex, ey, hx, hy, hz, sample_rate: float):
        self.ex = ex
        self.ey = ey
        self.hx = hx
        self.hy = hy
        self.hz = hz
        self.sample_rate = sample_rate

    def summary(self) -> dict:
        """返回统计摘要"""
        return {
            "n_samples": len(self.ex),
            "duration_s": len(self.ex) / self.sample_rate,
            "sample_rate_hz": self.sample_rate,
            "ex": {"min": self.ex.min(), "max": self.ex.max(), "std": self.ex.std()},
            "ey": {"min": self.ey.min(), "max": self.ey.max(), "std": self.ey.std()},
            "hx": {"min": self.hx.min(), "max": self.hx.max(), "std": self.hx.std()},
            "hy": {"min": self.hy.min(), "max": self.hy.max(), "std": self.hy.std()},
            "hz": {"min": self.hz.min(), "max": self.hz.max(), "std": self.hz.std()},
        }

    def print_summary(self):
        """打印统计摘要"""
        s = self.summary()
        print(f"    采样点数: {s['n_samples']}")
        print(f"    时长: {s['duration_s']:.1f} 秒")
        print(f"    采样率: {s['sample_rate_hz']} Hz")
        print(
            f"    Ex: [{s['ex']['min']:.6f}, {s['ex']['max']:.6f}] V/m, std={s['ex']['std']:.6f}"
        )
        print(
            f"    Ey: [{s['ey']['min']:.6f}, {s['ey']['max']:.6f}] V/m, std={s['ey']['std']:.6f}"
        )
        print(
            f"    Hx: [{s['hx']['min']:.9f}, {s['hx']['max']:.9f}] A/m, std={s['hx']['std']:.9f}"
        )
        print(
            f"    Hy: [{s['hy']['min']:.9f}, {s['hy']['max']:.9f}] A/m, std={s['hy']['std']:.9f}"
        )
        print(
            f"    Hz: [{s['hz']['min']:.9f}, {s['hz']['max']:.9f}] A/m, std={s['hz']['std']:.9f}"
        )


# ============================================================================
# 主程序
# ============================================================================


def run_example():
    """运行一维MT正演合成示例"""

    print("=" * 60)
    print("一维MT模型正演合成时间序列示例")
    print("=" * 60)

    # --- 步骤1: 定义1D模型 ---
    print("\n[1] 定义1D地电模型...")
    model = MT1DModel("均匀半空间", resistivity=[100.0])
    periods = np.logspace(-2, 2, 16)
    print(f"    模型: {model}")
    print(f"    周期范围: {periods.min():.2f} s ~ {periods.max():.2f} s")

    # --- 步骤2: 构建EMFields ---
    print("\n[2] 构建EM场数据...")
    builder = EMFieldsBuilder(model)
    fields = builder.build_fields(periods)
    print(f"    频点数: {len(fields)}")

    # 显示前几个频点
    print("    频率        周期      Zxy (视电阻率)")
    print("    " + "-" * 45)
    for f in fields[:5]:
        T = 1.0 / f.freq
        app_rho = abs(f.zxy) ** 2 * 1e6 / (2 * np.pi)
        print(f"    {f.freq:8.4f} Hz  {T:8.4f} s  {app_rho:10.2f} Ohm·m")
    print("    ...")

    # --- 步骤3: 创建测点 ---
    print("\n[3] 创建正演测点...")
    site = TimeSeriesGenerator.create_site("1D_Model_Site", 100.0, 200.0, fields)
    print(f"    测点: {site.name}, 位置=({site.x}, {site.y})")

    # --- 步骤4: 配置生成器 ---
    print("\n[4] 配置时间序列生成器...")
    schema = SyntheticSchema.from_ts("TS3")
    generator = TimeSeriesGenerator(schema)
    print(f"    系统: {schema.name}, 采样率={schema.sample_rate} Hz")
    print(f"    方法: {generator.method.name}")

    # --- 步骤5: 生成时间序列 ---
    print("\n[5] 生成时间序列...")
    t1 = datetime(2023, 6, 15, 10, 0, 0)
    t2 = datetime(2023, 6, 15, 10, 0, 10)
    ex, ey, hx, hy, hz = generator.generate(site, t1, t2, seed=42)
    print(f"    时间: {t1} ~ {t2}")

    # --- 步骤6: 统计信息 ---
    print("\n[6] 时间序列统计...")
    stats = TimeSeriesStats(ex, ey, hx, hy, hz, schema.sample_rate)
    stats.print_summary()

    # --- 步骤7: 保存结果 ---
    print("\n[7] 保存结果...")
    output_dir = "./output"
    os.makedirs(output_dir, exist_ok=True)

    gmt_file = save_gmt_timeseries(
        output_dir, "1D_MT_Example", ex, ey, hx, hy, hz, t1, schema.sample_rate
    )
    print(f"    GMT: {gmt_file}")

    csv_file = os.path.join(output_dir, "1D_MT_Example.csv")
    save_csv_timeseries(csv_file, ex, ey, hx, hy, hz)
    print(f"    CSV: {csv_file}")

    # --- 步骤8: 验证1D特征 ---
    print("\n[8] 验证1D模型特征...")
    validator = Model1DValidator(fields, (ex, ey, hx, hy, hz))
    results = validator.validate_all()

    checks = [
        ("阻抗反对称 (Zxy=-Zyx)", results["impedance_symmetry"]),
        ("对角元为零 (Zxx=Zyy=0)", results["zero_diagonal"]),
        ("Tipper为零", results["zero_tipper"]),
        ("垂直分量为零 (Hz=0)", results["zero_vertical"]),
    ]

    for name, result in checks:
        status = "[PASS]" if result["passed"] else "[FAIL]"
        print(f"    {status} {name}: {result['message']}")

    print(f"\n    验证结果: {'全部通过' if results['all_passed'] else '存在失败项'}")

    print("\n" + "=" * 60)
    print("完成!")
    print("=" * 60)

    return {
        "model": model,
        "site": site,
        "fields": fields,
        "time_series": (ex, ey, hx, hy, hz),
        "stats": stats.summary(),
        "validation": results,
    }


if __name__ == "__main__":
    results = run_example()
