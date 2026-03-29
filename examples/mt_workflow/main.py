"""
MT 1D正演合成与处理完整流程

主工作流:
1. 定义1D地电模型
2. 正演计算阻抗和电磁场
3. 多频段时间序列合成
4. FFT频谱分析
5. 从合成数据反算阻抗
6. 视电阻率相位对比
7. 1D特征验证

用法:
    python -m mt_workflow.main
"""

import numpy as np
import os
import sys
from datetime import datetime
from typing import Dict, Tuple, Optional

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from .config import (
    TS_CONFIGS,
    get_config,
    TSConfig,
    MULTI_BAND_CONFIG,
    MU0,
    get_default_forward_periods,
    get_default_processing_periods,
    FORWARD_PERIODS_CONFIG,
    PROCESSING_PERIODS_CONFIG,
)
from .model_1d import MT1DModel, create_uniform_halfspace, get_preset_model
from .forward_1d import MT1DForward, compute_theoretical_response
from .synthesizer import TimeSeriesSynthesizer, create_site_for_periods
from .processor import TimeSeriesProcessor
from .validator import Model1DValidator, ResultsComparator, print_model_summary

# 导入synthetic_mt的EMFields和ForwardSite
from synthetic_mt import ForwardSite


def run_single_band_workflow(
    band: str, model: MT1DModel, periods: np.ndarray, duration: float, seed: int = 42
) -> Dict:
    """
    运行单频段工作流

    Args:
        band: 频段名称 ('TS3', 'TS4', 'TS5')
        model: 1D地电模型
        periods: 周期数组
        duration: 时间序列时长 (秒)
        seed: 随机种子

    Returns:
        结果字典
    """
    print(f"\n{'=' * 70}")
    print(f"频段: {band}")
    print(f"{'=' * 70}")

    # 1. 正演计算
    print("\n[1] 正演计算...")
    forward = MT1DForward(model)
    impedance = forward.calculate_impedance(periods)
    rho_a_theory, phase_theory = forward.calculate_app_resistivity_phase(periods)
    fields = forward.calculate_fields(periods)

    print(f"    周期范围: {periods.min():.4f} s ~ {periods.max():.2f} s")
    print(f"    频点数: {len(periods)}")

    # 2. 创建测点
    print("\n[2] 创建测点...")
    site = ForwardSite(name=f"{model.name}_{band}", x=0.0, y=0.0, fields=fields)
    print(f"    测点: {site.name}")

    # 3. 合成时间序列
    print("\n[3] 合成时间序列...")
    config = get_config(band)
    synth = TimeSeriesSynthesizer(config)

    t1 = datetime(2023, 1, 1, 0, 0, 0)
    t2 = datetime(2023, 1, 1, 0, 0, int(duration))

    ex, ey, hx, hy, hz = synth.generate(site, t1, t2, seed=seed)

    print(f"    配置: {config.name}, {config.sample_rate} Hz")
    print(f"    时长: {duration} s")
    print(f"    采样点: {len(ex)}")

    # 4. 时间序列处理
    print("\n[4] 时间序列处理 (FFT)...")
    processor = TimeSeriesProcessor(ex, ey, hx, hy, hz, config.sample_rate)

    # 估算阻抗
    est_result = processor.estimate_impedance_at_periods(periods)

    # 从估算结果提取阻抗 (避免重新计算)
    zxy_est = est_result["Zxy"]
    zyx_est = est_result["Zyx"]
    rho_a_est = est_result["app_resistivity"]
    phase_est = est_result["phase"]

    print(f"    频率分辨率: {processor.get_frequency_resolution():.4f} Hz")

    # 5. 结果对比
    print("\n[5] 结果对比...")
    comparator = ResultsComparator(
        periods, rho_a_theory, phase_theory, rho_a_est, phase_est
    )
    comparator.print_comparison(max_rows=10)

    # 6. 验证1D特征
    print("\n[6] 验证1D特征...")
    validator = Model1DValidator(fields, (ex, ey, hx, hy, hz))
    for result in validator.validate_all():
        status = "[PASS]" if result.passed else "[FAIL]"
        print(f"    {status} {result.name}: {result.message}")

    return {
        "band": band,
        "config": config,
        "model": model,
        "periods": periods,
        "impedance": impedance,
        "fields": fields,
        "time_series": (ex, ey, hx, hy, hz),
        "rho_a_theory": rho_a_theory,
        "phase_theory": phase_theory,
        "rho_a_est": rho_a_est,
        "phase_est": phase_est,
        "processor": processor,
        "validator": validator,
    }


def run_multi_band_workflow(
    model: MT1DModel, bands: Dict[str, Dict] = None, seeds: Dict[str, int] = None
) -> Dict:
    """
    运行多频段工作流

    Args:
        model: 1D地电模型
        bands: 频段配置 {'TS3': {'period_range': (0.001, 1.0), 'duration': 10}, ...}
        seeds: 随机种子

    Returns:
        各频段结果字典
    """
    if bands is None:
        bands = MULTI_BAND_CONFIG

    if seeds is None:
        seeds = {"TS3": 42, "TS4": 43, "TS5": 44}

    print("\n" + "=" * 70)
    print(f"多频段工作流: {model.name}")
    print("=" * 70)

    results = {}
    for band, config_b in bands.items():
        # 生成周期数组
        period_min, period_max = config_b["period_range"]
        n_periods = config_b.get("n_periods", 8)
        periods = np.logspace(np.log10(period_min), np.log10(period_max), n_periods)

        # 运行单频段
        results[band] = run_single_band_workflow(
            band=band,
            model=model,
            periods=periods,
            duration=config_b["duration"],
            seed=seeds.get(band, 42),
        )

    return results


def run_simple_workflow():
    """运行简化工作流 (单频段演示)"""

    print("=" * 70)
    print("1D MT 正演合成与处理完整流程")
    print("=" * 70)

    # 1. 定义模型
    print("\n[步骤1] 定义1D地电模型...")
    model = create_uniform_halfspace("uniform_100", rho=100.0)
    print(f"    模型: {model}")

    # 2. 定义周期范围 - 使用默认正演合成频率
    # 频率: 1000Hz ~ 100000s，每个数量级20个频点，总共161个频点
    periods = get_default_forward_periods()
    print(f"    正演合成周期范围: {periods.min():.6f} s ~ {periods.max():.0f} s")
    print(
        f"    正演合成频点数: {len(periods)} (每数量级{FORWARD_PERIODS_CONFIG['points_per_decade']}个)"
    )

    # 3. 正演计算
    print("\n[步骤2] 正演计算...")
    forward = MT1DForward(model)
    impedance = forward.calculate_impedance(periods)
    rho_a_theory, phase_theory = forward.calculate_app_resistivity_phase(periods)

    # 打印阻抗 (显示部分)
    print("\n    正演阻抗 (部分频点):")
    print(f"    {'T(s)':<12} {'f(Hz)':<10} {'rho_a(Ohm·m)':<15} {'phase(deg)':<12}")
    print(f"    {'-' * 52}")
    # 显示8个有代表性的频点
    indices = [0, 20, 40, 60, 80, 100, 120, 140, 160]
    for idx in indices:
        if idx < len(periods):
            f = 1.0 / periods[idx]
            print(
                f"    {periods[idx]:<12.4f} {f:<10.4f} {rho_a_theory[idx]:<15.2f} {phase_theory[idx]:<12.1f}"
            )
    print("    ...")

    # 4. 计算电磁场
    print("\n[步骤3] 计算两个极化的4分量电磁场...")
    fields = forward.calculate_fields(periods)

    print("\n    极化1 (TM模式): Hy主, Hx小")
    print(f"    {'T(s)':<8} {'Ex1':<12} {'Ey1':<12} {'Hx1':<12} {'Hy1':<12}")
    print(f"    {'-' * 56}")
    for i in range(min(5, len(fields))):
        f = fields[i]
        T = 1.0 / f.freq
        print(
            f"    {T:<8.4f} {abs(f.ex1):<12.6f} {abs(f.ey1):<12.6f} {abs(f.hx1):<12.6f} {abs(f.hy1):<12.6f}"
        )
    print("    ...")

    print("\n    极化2 (TE模式): Hx主, Hy小")
    print(f"    {'T(s)':<8} {'Ex2':<12} {'Ey2':<12} {'Hx2':<12} {'Hy2':<12}")
    print(f"    {'-' * 56}")
    for i in range(min(5, len(fields))):
        f = fields[i]
        T = 1.0 / f.freq
        print(
            f"    {T:<8.4f} {abs(f.ex2):<12.6f} {abs(f.ey2):<12.6f} {abs(f.hx2):<12.6f} {abs(f.hy2):<12.6f}"
        )
    print("    ...")

    # 5. 创建测点
    print("\n[步骤4] 创建测点...")
    site = ForwardSite(name="TestSite", x=0, y=0, fields=fields)
    print(f"    测点: {site.name}")
    print(f"    频点数: {len(site.fields)}")

    # 6. 合成时间序列
    print("\n[步骤5] 合成时间序列...")
    config = get_config("TS3")  # 使用TS3: 2400Hz
    synth = TimeSeriesSynthesizer(config)

    duration = 10.0  # 10秒
    t1 = datetime(2023, 1, 1, 0, 0, 0)
    t2 = datetime(2023, 1, 1, 0, 0, int(duration))

    ex, ey, hx, hy, hz = synth.generate(site, t1, t2, seed=42)

    print(f"    配置: {config.name}, {config.sample_rate} Hz")
    print(f"    时长: {duration} s")
    print(f"    采样点: {len(ex)}")

    # 统计
    print("\n    时间序列统计:")
    print(f"    Ex: [{ex.min():.6f}, {ex.max():.6f}] V/m")
    print(f"    Ey: [{ey.min():.6f}, {ey.max():.6f}] V/m")
    print(f"    Hx: [{hx.min():.9f}, {hx.max():.9f}] A/m")
    print(f"    Hy: [{hy.min():.9f}, {hy.max():.9f}] A/m")
    print(f"    Hz: [{hz.min():.9f}, {hz.max():.9f}] A/m")

    # 7. FFT处理
    print("\n[步骤6] FFT频谱分析...")
    processor = TimeSeriesProcessor(ex, ey, hx, hy, hz, config.sample_rate)

    # 估算阻抗
    est_result = processor.estimate_impedance_at_periods(periods)

    print(f"    频率分辨率: {processor.get_frequency_resolution():.4f} Hz")
    print(
        f"    可分析频率范围: {processor.get_frequency_resolution():.4f} Hz ~ {config.sample_rate / 2:.0f} Hz"
    )

    # 8. 反演对比
    print("\n[步骤7] 反演结果对比...")

    comparator = ResultsComparator(
        periods,
        rho_a_theory,
        phase_theory,
        est_result["app_resistivity"],
        est_result["phase"],
    )
    comparator.print_comparison(max_rows=10)

    # 9. 验证
    print("\n[步骤8] 1D特征验证...")
    validator = Model1DValidator(fields, (ex, ey, hx, hy, hz))
    all_passed = True
    for result in validator.validate_all():
        status = "[PASS]" if result.passed else "[FAIL]"
        if not result.passed:
            all_passed = False
        print(f"    {status} {result.name}: {result.message}")

    print("\n" + "=" * 70)
    print("工作流完成!")
    print("=" * 70)

    return {
        "model": model,
        "periods": periods,
        "fields": fields,
        "time_series": (ex, ey, hx, hy, hz),
        "rho_a_theory": rho_a_theory,
        "phase_theory": phase_theory,
        "rho_a_est": est_result["app_resistivity"],
        "phase_est": est_result["phase"],
    }


def main():
    """主入口"""
    import argparse

    parser = argparse.ArgumentParser(description="1D MT 正演合成与处理流程")
    parser.add_argument(
        "--band",
        choices=["TS3", "TS4", "TS5", "multi"],
        default="TS3",
        help="频段 (默认: TS3)",
    )
    parser.add_argument(
        "--model",
        default="uniform_100",
        choices=["uniform_100", "uniform_1000", "two_layer_ll", "two_layer_hl"],
        help="地电模型",
    )
    parser.add_argument(
        "--duration", type=float, default=10.0, help="时间序列时长 (秒)"
    )
    parser.add_argument("--seed", type=int, default=42, help="随机种子")

    args = parser.parse_args()

    # 获取模型
    model = get_preset_model(args.model)
    if model is None:
        model = create_uniform_halfspace(args.model, rho=100.0)

    if args.band == "multi":
        # 多频段
        run_multi_band_workflow(model)
    else:
        # 单频段
        config = get_config(args.band)

        # 生成周期
        period_min = max(config.period_min, 0.001)
        period_max = min(config.period_max, 1000)
        periods = np.logspace(np.log10(period_min), np.log10(period_max), 12)

        run_single_band_workflow(args.band, model, periods, args.duration, args.seed)


if __name__ == "__main__":
    run_simple_workflow()
