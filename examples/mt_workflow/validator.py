"""
结果验证与对比模块

验证1D模型特征，对比正演与反演结果
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """验证结果"""

    passed: bool
    name: str
    message: str
    value: float
    expected: float
    tolerance: float


class Model1DValidator:
    """
    1D模型特征验证器

    验证:
    - Zxx ≈ 0
    - Zyy ≈ 0
    - Zxy ≈ -Zyx
    - Tipper ≈ 0
    - Hz ≈ 0 (时间序列)
    """

    def __init__(self, fields: List = None, time_series: Tuple[np.ndarray, ...] = None):
        """
        Args:
            fields: EMFields列表
            time_series: (ex, ey, hx, hy, hz)元组
        """
        self.fields = fields
        self.ex, self.ey, self.hx, self.hy, self.hz = (
            time_series if time_series else (None,) * 5
        )

    def check_impedance_symmetry(self) -> ValidationResult:
        """检查阻抗反对称性: Zxy = -Zyx"""
        if self.fields is None:
            return ValidationResult(
                False, "impedance_symmetry", "No fields provided", 0, 0, 0
            )

        zxy_mag = np.mean([abs(f.zxy) for f in self.fields])
        zyx_mag = np.mean([abs(f.zyx) for f in self.fields])

        if zyx_mag < 1e-20:
            ratio = 1.0
        else:
            ratio = zxy_mag / zyx_mag

        expected_ratio = 1.0
        tolerance = 0.05  # 5%
        passed = abs(ratio - expected_ratio) < tolerance

        return ValidationResult(
            passed=passed,
            name="impedance_symmetry",
            message=f"Zxy/Zyx = {ratio:.4f} (expected {expected_ratio})",
            value=ratio,
            expected=expected_ratio,
            tolerance=tolerance,
        )

    def check_zero_diagonal(self) -> ValidationResult:
        """检查阻抗对角元为零: Zxx = Zyy = 0"""
        if self.fields is None:
            return ValidationResult(
                False, "zero_diagonal", "No fields provided", 0, 0, 0
            )

        zxx_max = max([abs(f.zxx) for f in self.fields]) if self.fields else 0
        zyy_max = max([abs(f.zyy) for f in self.fields]) if self.fields else 0

        max_val = max(zxx_max, zyy_max)
        tolerance = 1e-10
        passed = max_val < tolerance

        return ValidationResult(
            passed=passed,
            name="zero_diagonal",
            message=f"Zxx_max={zxx_max:.2e}, Zyy_max={zyy_max:.2e}",
            value=max_val,
            expected=0,
            tolerance=tolerance,
        )

    def check_zero_tipper(self) -> ValidationResult:
        """检查Tipper为零"""
        if self.fields is None:
            return ValidationResult(False, "zero_tipper", "No fields provided", 0, 0, 0)

        tzx_max = max([abs(f.tzx) for f in self.fields]) if self.fields else 0
        tzy_max = max([abs(f.tzy) for f in self.fields]) if self.fields else 0

        max_val = max(tzx_max, tzy_max)
        tolerance = 1e-10
        passed = max_val < tolerance

        return ValidationResult(
            passed=passed,
            name="zero_tipper",
            message=f"Tzx_max={tzx_max:.2e}, Tzy_max={tzy_max:.2e}",
            value=max_val,
            expected=0,
            tolerance=tolerance,
        )

    def check_zero_vertical_component(self) -> ValidationResult:
        """检查垂直分量Hz ≈ 0"""
        if self.hz is None:
            return ValidationResult(
                False, "zero_vertical", "No time series provided", 0, 0, 0
            )

        hz_max = np.max(np.abs(self.hz))
        hxy_max = max(np.max(np.abs(self.hx)), np.max(np.abs(self.hy)))

        if hxy_max < 1e-20:
            ratio = 0
        else:
            ratio = hz_max / hxy_max

        tolerance = 1e-6
        passed = ratio < tolerance

        return ValidationResult(
            passed=passed,
            name="zero_vertical",
            message=f"Hz_max/Hxy_max = {ratio:.2e}",
            value=ratio,
            expected=0,
            tolerance=tolerance,
        )

    def validate_all(self) -> List[ValidationResult]:
        """执行所有验证"""
        results = [
            self.check_impedance_symmetry(),
            self.check_zero_diagonal(),
            self.check_zero_tipper(),
        ]

        if self.hz is not None:
            results.append(self.check_zero_vertical_component())

        return results


class ResultsComparator:
    """
    正演与反演结果对比器
    """

    def __init__(
        self,
        periods: np.ndarray,
        forward_rho: np.ndarray,
        forward_phase: np.ndarray,
        estimated_rho: np.ndarray = None,
        estimated_phase: np.ndarray = None,
    ):
        """
        Args:
            periods: 周期数组
            forward_rho: 正演视电阻率
            forward_phase: 正演相位
            estimated_rho: 反演视电阻率 (可选)
            estimated_phase: 反演相位 (可选)
        """
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

        error = np.abs(self.forward_phase - self.estimated_phase)
        return error

    def get_comparison_table(self, max_rows: int = 20) -> List[Dict]:
        """获取对比表格数据"""
        rows = []
        n = min(len(self.periods), max_rows)

        for i in range(n):
            row = {
                "period": self.periods[i],
                "freq": 1.0 / self.periods[i],
                "rho_forward": self.forward_rho[i],
                "phase_forward": self.forward_phase[i],
            }

            if self.estimated_rho is not None:
                row["rho_estimated"] = self.estimated_rho[i]
                row["rho_error"] = (
                    self.compute_rho_error()[i]
                    if self.compute_rho_error() is not None
                    else None
                )

            if self.estimated_phase is not None:
                row["phase_estimated"] = self.estimated_phase[i]
                row["phase_error"] = (
                    self.compute_phase_error()[i]
                    if self.compute_phase_error() is not None
                    else None
                )

            rows.append(row)

        return rows

    def print_comparison(self, max_rows: int = 16):
        """打印对比结果"""
        print("\n" + "=" * 80)
        print("正演 vs 反演对比")
        print("=" * 80)

        if self.estimated_rho is not None:
            print(
                f"\n{'T(s)':<10} {'rho_a真':<12} {'rho_a反':<12} {'误差%':<10} {'pha真':<10} {'pha反':<10} {'相位误差':<10}"
            )
            print("-" * 80)

            for row in self.get_comparison_table(max_rows):
                T = row["period"]
                rho_t = row["rho_forward"]
                rho_e = row.get("rho_estimated", "N/A")
                err_rho = row.get("rho_error", "N/A")
                pha_t = row["phase_forward"]
                pha_e = row.get("phase_estimated", "N/A")
                err_pha = row.get("phase_error", "N/A")

                if isinstance(err_rho, float):
                    err_rho_str = f"{err_rho:.1f}"
                else:
                    err_rho_str = str(err_rho)

                if isinstance(err_pha, float):
                    err_pha_str = f"{err_pha:.1f}"
                else:
                    err_pha_str = str(err_pha)

                if isinstance(rho_e, float):
                    rho_e_str = f"{rho_e:.2f}"
                else:
                    rho_e_str = str(rho_e)

                if isinstance(pha_e, float):
                    pha_e_str = f"{pha_e:.1f}"
                else:
                    pha_e_str = str(pha_e)

                print(
                    f"{T:<10.4f} {rho_t:<12.2f} {rho_e_str:<12} {err_rho_str:<10} {pha_t:<10.1f} {pha_e_str:<10} {err_pha_str:<10}"
                )
        else:
            print(f"\n{'T(s)':<10} {'rho_a(Ohm·m)':<15} {'phase(deg)':<12}")
            print("-" * 40)
            for i in range(min(len(self.periods), max_rows)):
                print(
                    f"{self.periods[i]:<10.4f} {self.forward_rho[i]:<15.2f} {self.forward_phase[i]:<12.1f}"
                )

        if len(self.periods) > max_rows:
            print("  ...")


def print_model_summary(
    model: "MT1DModel",
    periods: np.ndarray,
    impedance: Dict,
    rho_a: np.ndarray,
    phase: np.ndarray,
):
    """打印模型摘要"""
    print("\n" + "=" * 70)
    print(f"1D地电模型: {model.name}")
    print("=" * 70)
    print(f"  层数: {model.n_layers}")

    if model.is_halfspace:
        print(f"  类型: 均匀半空间")
        print(f"  电阻率: {model.resistivity[0]:.1f} Ohm·m")
    else:
        print(f"  类型: 层状模型")
        for i, (rho, h) in enumerate(zip(model.resistivity, model.thickness)):
            print(f"    层{i + 1}: rho = {rho:.1f} Ohm·m, 厚度 = {h:.1f} m")
        print(f"    半空间: rho = {model.resistivity[-1]:.1f} Ohm·m")

    print(f"\n  正演周期范围: {periods.min():.4f} s ~ {periods.max():.2f} s")
    print(f"  频点数量: {len(periods)}")

    print("\n  理论视电阻率和相位:")
    print(f"  {'T(s)':<10} {'rho_a(Ohm·m)':<15} {'phase(deg)':<12}")
    print(f"  {'-' * 40}")
    for i in range(min(len(periods), 8)):
        print(f"  {periods[i]:<10.4f} {rho_a[i]:<15.2f} {phase[i]:<12.1f}")
    if len(periods) > 8:
        print("  ...")
