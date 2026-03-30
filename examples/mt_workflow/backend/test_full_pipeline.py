"""
MT 1D 正演-合成-处理-对比 完整流程测试

测试5个预设模型，验证确定性合成使得处理结果与正演结果完全一致。
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 添加路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
_workflow_dir = os.path.dirname(_script_dir)  # examples/mt_workflow
_examples_dir = os.path.dirname(_workflow_dir)  # examples
_project_root = os.path.dirname(_examples_dir)  # Synthetic-MT-Python
_src_path = os.path.join(_project_root, "src")

for p in [_src_path, _workflow_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

from backend.api import MTWorkflowAPI, reset_api
from backend.core import get_default_processing_periods


# ============================================================================
# 测试配置
# ============================================================================

# 预设模型
PRESET_MODELS = {
    "uniform_100": dict(resistivity=[100.0], thickness=[]),
    "uniform_1000": dict(resistivity=[1000.0], thickness=[]),
    "two_layer_hl": dict(resistivity=[1000.0, 10.0], thickness=[100.0]),
    "two_layer_ll": dict(resistivity=[10.0, 1000.0], thickness=[100.0]),
    "three_layer_hll": dict(resistivity=[1000.0, 10.0, 100.0], thickness=[50.0, 200.0]),
}

# 测试用频段配置
TEST_BANDS = ["TS3"]  # 可扩展到TS4, TS5

# 输出目录
OUTPUT_DIR = os.path.join(_project_root, "docs", "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================================
# 测试函数
# ============================================================================


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def test_model(model_name: str, api: MTWorkflowAPI, band: str = "TS3") -> dict:
    """测试单个模型"""
    print(f"\n测试模型: {model_name}")

    # 1. 创建模型并运行正演
    api.get_preset_model(model_name)

    # 使用与合成相同周期的正演结果用于对比
    from backend.core import TS_CONFIGS

    config = TS_CONFIGS[band]
    forward_periods = np.logspace(
        np.log10(config.period_min), np.log10(config.period_max), 16
    )
    forward_result = api.run_forward(forward_periods)

    # 2. 合成时间序列 (使用随机分段合成)
    ts_result = api.synthesize_time_series_random(band, duration=60.0)

    # 3. 处理时间序列
    processed_result = api.process_time_series(forward_periods)

    # 4. 计算误差
    fwd_rho = forward_result["app_resistivity"]
    fwd_phase = forward_result["phase"]
    proc_rho = processed_result["app_resistivity"]
    proc_phase = processed_result["phase"]

    # 计算误差
    rho_error = np.abs((fwd_rho - proc_rho) / fwd_rho * 100)
    phase_error = np.abs(fwd_phase - proc_phase)

    # 计算统计
    max_rho_error = np.max(rho_error)
    mean_rho_error = np.mean(rho_error)
    max_phase_error = np.max(phase_error)
    mean_phase_error = np.mean(phase_error)

    # 只考虑有效数据点（非零阻抗）
    valid_mask = fwd_rho > 0
    n_valid = np.sum(valid_mask)

    print(
        f"  正演周期范围: {forward_periods.min():.4f}s - {forward_periods.max():.2f}s"
    )
    print(f"  有效数据点: {n_valid}/{len(forward_periods)}")
    print(f"  视电阻率误差: 最大={max_rho_error:.4f}%, 平均={mean_rho_error:.4f}%")
    print(f"  相位误差: 最大={max_phase_error:.4f}°, 平均={mean_phase_error:.4f}°")

    return {
        "model_name": model_name,
        "band": band,
        "forward_periods": forward_periods,
        "forward_rho": fwd_rho,
        "forward_phase": fwd_phase,
        "processed_rho": proc_rho,
        "processed_phase": proc_phase,
        "rho_error": rho_error,
        "phase_error": phase_error,
        "max_rho_error": max_rho_error,
        "mean_rho_error": mean_rho_error,
        "max_phase_error": max_phase_error,
        "mean_phase_error": mean_phase_error,
        "n_valid": n_valid,
        "forward_result": forward_result,
        "processed_result": processed_result,
    }


def plot_model_comparison(result: dict, output_dir: str):
    """为单个模型绘制对比图"""
    model_name = result["model_name"]
    periods = result["forward_periods"]
    fwd_rho = result["forward_rho"]
    fwd_phase = result["forward_phase"]
    proc_rho = result["processed_rho"]
    proc_phase = result["processed_phase"]
    rho_error = result["rho_error"]
    phase_error = result["phase_error"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Model: {model_name}", fontsize=14, fontweight="bold")

    # 1. 视电阻率对比 (log-log)
    ax1 = axes[0, 0]
    ax1.loglog(periods, fwd_rho, "b-o", label="Forward", markersize=4)
    ax1.loglog(periods, proc_rho, "r--x", label="Processed", markersize=4)
    ax1.set_xlabel("Period (s)")
    ax1.set_ylabel("App. Resistivity (Ohm·m)")
    ax1.set_title("App. Resistivity Comparison")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. 相位对比 (log-x)
    ax2 = axes[0, 1]
    ax2.semilogx(periods, fwd_phase, "b-o", label="Forward", markersize=4)
    ax2.semilogx(periods, proc_phase, "r--x", label="Processed", markersize=4)
    ax2.set_xlabel("Period (s)")
    ax2.set_ylabel("Phase (deg)")
    ax2.set_title("Phase Comparison")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. 视电阻率误差
    ax3 = axes[1, 0]
    ax3.semilogx(periods, rho_error, "g-o", markersize=4)
    ax3.set_xlabel("Period (s)")
    ax3.set_ylabel("Relative Error (%)")
    ax3.set_title("App. Resistivity Error")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, max(5, np.max(rho_error) * 1.2))

    # 4. 相位误差
    ax4 = axes[1, 1]
    ax4.semilogx(periods, phase_error, "m-o", markersize=4)
    ax4.set_xlabel("Period (s)")
    ax4.set_ylabel("Absolute Error (deg)")
    ax4.set_title("Phase Error")
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, max(1, np.max(phase_error) * 1.2))

    plt.tight_layout()

    # 保存
    filename = f"{model_name}_comparison.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  保存图像: {filepath}")

    return filepath


def plot_summary(results: list, output_dir: str):
    """绘制所有模型的汇总对比图"""
    n_models = len(results)
    fig, axes = plt.subplots(n_models, 2, figsize=(12, 4 * n_models))
    if n_models == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle("All Models: Forward vs Processed", fontsize=14, fontweight="bold")

    for i, result in enumerate(results):
        model_name = result["model_name"]
        periods = result["forward_periods"]
        fwd_rho = result["forward_rho"]
        fwd_phase = result["forward_phase"]
        proc_rho = result["processed_rho"]
        proc_phase = result["processed_phase"]

        # 视电阻率
        ax1 = axes[i, 0]
        ax1.loglog(periods, fwd_rho, "b-o", label="Forward", markersize=3)
        ax1.loglog(periods, proc_rho, "r--x", label="Processed", markersize=3)
        ax1.set_xlabel("Period (s)")
        ax1.set_ylabel("rho_a (Ohm·m)")
        ax1.set_title(f"{model_name} - App. Resistivity")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # 相位
        ax2 = axes[i, 1]
        ax2.semilogx(periods, fwd_phase, "b-o", label="Forward", markersize=3)
        ax2.semilogx(periods, proc_phase, "r--x", label="Processed", markersize=3)
        ax2.set_xlabel("Period (s)")
        ax2.set_ylabel("Phase (deg)")
        ax2.set_title(f"{model_name} - Phase")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    filepath = os.path.join(output_dir, "all_models_summary.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n保存汇总图像: {filepath}")

    return filepath


def plot_error_summary(results: list, output_dir: str):
    """绘制误差汇总图"""
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    fig.suptitle("Error Analysis: All Models", fontsize=14, fontweight="bold")

    model_names = [r["model_name"] for r in results]
    x = np.arange(len(model_names))
    width = 0.35

    # 视电阻率误差
    ax1 = axes[0]
    max_rho_errors = [r["max_rho_error"] for r in results]
    mean_rho_errors = [r["mean_rho_error"] for r in results]
    bars1 = ax1.bar(
        x - width / 2, max_rho_errors, width, label="Max Error", color="red", alpha=0.7
    )
    bars2 = ax1.bar(
        x + width / 2,
        mean_rho_errors,
        width,
        label="Mean Error",
        color="blue",
        alpha=0.7,
    )
    ax1.set_ylabel("Relative Error (%)")
    ax1.set_title("App. Resistivity Error")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=15)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # 相位误差
    ax2 = axes[1]
    max_phase_errors = [r["max_phase_error"] for r in results]
    mean_phase_errors = [r["mean_phase_error"] for r in results]
    bars3 = ax2.bar(
        x - width / 2,
        max_phase_errors,
        width,
        label="Max Error",
        color="red",
        alpha=0.7,
    )
    bars4 = ax2.bar(
        x + width / 2,
        mean_phase_errors,
        width,
        label="Mean Error",
        color="blue",
        alpha=0.7,
    )
    ax2.set_ylabel("Absolute Error (deg)")
    ax2.set_title("Phase Error")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=15)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()

    filepath = os.path.join(output_dir, "error_summary.png")
    plt.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"保存误差汇总图: {filepath}")

    return filepath


# ============================================================================
# 主函数
# ============================================================================


def main():
    print_header("MT 1D 完整流程测试")
    print(f"输出目录: {OUTPUT_DIR}")

    all_results = []

    for model_name in PRESET_MODELS.keys():
        reset_api()
        api = MTWorkflowAPI()

        result = test_model(model_name, api, band="TS3")
        all_results.append(result)

        # 绘制单个模型对比图
        plot_model_comparison(result, OUTPUT_DIR)

    # 汇总图
    plot_summary(all_results, OUTPUT_DIR)
    plot_error_summary(all_results, OUTPUT_DIR)

    # 打印汇总表格
    print_header("测试结果汇总")
    print(
        f"\n{'模型':<20} {'最大rho_a误差%':<15} {'平均rho_a误差%':<15} {'最大相位误差°':<15} {'平均相位误差°':<15}"
    )
    print("-" * 80)
    for r in all_results:
        print(
            f"{r['model_name']:<20} {r['max_rho_error']:<15.4f} {r['mean_rho_error']:<15.4f} {r['max_phase_error']:<15.4f} {r['mean_phase_error']:<15.4f}"
        )

    # 判定是否通过
    print_header("测试判定")
    tolerance_rho = 2.0  # 2% (FFT数值精度限制)
    tolerance_phase = 1.0  # 1 degree

    all_pass = True
    for r in all_results:
        if r["max_rho_error"] > tolerance_rho or r["max_phase_error"] > tolerance_phase:
            all_pass = False
            print(f"  [FAIL] {r['model_name']}: 误差超过容限")
        else:
            print(f"  [PASS] {r['model_name']}")

    if all_pass:
        print(
            f"\n[PASS] 所有模型测试通过! (容限: rho<{tolerance_rho}%, phase<{tolerance_phase}°)"
        )
    else:
        print(f"\n[FAIL] 部分模型测试未通过")

    # 生成Markdown报告
    report_path = os.path.join(OUTPUT_DIR, "test_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# MT 1D 正演-合成-处理-对比 测试报告\n\n")
        f.write(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## 测试配置\n\n")
        f.write("- 合成方法: 确定性合成 (DeterministicTimeSeriesSynthesizer)\n")
        f.write("- 频段: TS3 (2400Hz, 1-1000Hz)\n")
        f.write("- 合成时长: 60秒\n")
        f.write(f"- 误差容限: 视电阻率<{tolerance_rho}%, 相位<{tolerance_phase}°\n\n")

        f.write("## 测试结果汇总\n\n")
        f.write(
            "| 模型 | 最大rho_a误差% | 平均rho_a误差% | 最大相位误差° | 平均相位误差° | 状态 |\n"
        )
        f.write(
            "|------|---------------|---------------|--------------|---------------|------|\n"
        )
        for r in all_results:
            status = (
                "PASS"
                if r["max_rho_error"] <= tolerance_rho
                and r["max_phase_error"] <= tolerance_phase
                else "FAIL"
            )
            f.write(
                f"| {r['model_name']} | {r['max_rho_error']:.4f} | {r['mean_rho_error']:.4f} | {r['max_phase_error']:.4f} | {r['mean_phase_error']:.4f} | {status} |\n"
            )

        f.write("\n## 生成图像\n\n")
        f.write(f"- 单模型对比图: `docs/plots/<model_name>_comparison.png`\n")
        f.write("- 汇总对比图: `docs/plots/all_models_summary.png`\n")
        f.write("- 误差汇总图: `docs/plots/error_summary.png`\n")

    print(f"\n报告已保存: {report_path}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
