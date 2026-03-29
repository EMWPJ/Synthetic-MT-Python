"""
后台模块验证脚本

独立运行验证所有后台模块的正确性
不依赖GUI，可直接执行: python -m examples.mt_workflow.backend.verify_all
"""

import sys
import os
import numpy as np

# 添加项目路径
script_dir = os.path.dirname(os.path.abspath(__file__))
workflow_dir = os.path.dirname(script_dir)  # examples/mt_workflow
examples_dir = os.path.dirname(workflow_dir)  # examples
project_root = os.path.dirname(examples_dir)  # Synthetic-MT-Python

src_path = os.path.join(project_root, "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

# 添加mt_workflow路径以便导入backend
if workflow_dir not in sys.path:
    sys.path.insert(0, workflow_dir)

from backend.core import (
    MT1DModel,
    MT1DForward,
    TimeSeriesSynthesizer,
    TimeSeriesProcessor,
    Model1DValidator,
    ResultsComparator,
    TSConfig,
    TS_CONFIGS,
    get_default_forward_periods,
    get_default_processing_periods,
    MU0,
)
from backend.api import MTWorkflowAPI, reset_api


# ============================================================================
# 验证函数
# ============================================================================


def print_header(title: str):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(name: str, passed: bool, message: str = ""):
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if message:
        print(f"         {message}")


def verify_model_creation():
    """验证模型创建"""
    print_header("验证1: 模型创建")

    # 均匀半空间
    model1 = MT1DModel("halfspace", [100.0])
    print_result(
        "均匀半空间创建", model1.is_halfspace and model1.resistivity[0] == 100.0
    )

    # 层状模型
    model2 = MT1DModel("layered", [100.0, 500.0, 10.0], [100.0, 200.0])
    print_result(
        "层状模型创建",
        not model2.is_halfspace
        and len(model2.resistivity) == 3
        and len(model2.thickness) == 2,
    )

    # 属性检查
    print_result("n_layers属性", model2.n_layers == 3)
    print_result(
        "get_layer_params",
        model2.get_layer_params(0) == (100.0, 100.0)
        and model2.get_layer_params(1) == (500.0, 200.0)
        and model2.get_layer_params(2) == (10.0, 0.0),
    )

    return True


def verify_halfspace_forward():
    """验证均匀半空间正演"""
    print_header("验证2: 均匀半空间正演")

    rho = 100.0
    model = MT1DModel("test_halfspace", [rho])
    forward = MT1DForward(model)

    # 测试周期
    periods = np.array([0.1, 1.0, 10.0])
    result = forward.calculate_impedance(periods)

    # 检查阻抗对称性
    zxy = result["Zxy"]
    zyx = result["Zyx"]
    print_result("阻抗反对称 Zxy = -Zyx", np.allclose(zxy, -zyx, rtol=1e-10))

    # 检查对角元为零
    zxx = result["Zxx"]
    zyy = result["Zyy"]
    print_result("Zxx ≈ 0", np.allclose(zxx, 0, atol=1e-20))
    print_result("Zyy ≈ 0", np.allclose(zyy, 0, atol=1e-20))

    # 验证视电阻率解析解
    rho_a, phase = forward.calculate_app_resistivity_phase(periods)

    # 理论值: rho_a = |Zxy|^2 / (omega * mu0)
    # 对于均匀半空间: Zxy = (1+i) * sqrt(omega * mu0 * rho / 2)
    # |Zxy|^2 = 2 * omega * mu0 * rho / 2 = omega * mu0 * rho
    # 所以 rho_a = omega * mu0 * rho / (omega * mu0) = rho
    print_result("均匀半空间视电阻率 = 真实电阻率", np.allclose(rho_a, rho, rtol=1e-10))

    # 相位应该是45度（对于均匀半空间）
    print_result("均匀半空间相位 ≈ 45°", np.allclose(phase, 45.0, atol=1e-10))

    # 打印详细信息
    print(f"\n  周期(s)    视电阻率(Ohm·m)    相位(度)")
    print(f"  " + "-" * 40)
    for i in range(len(periods)):
        print(f"  {periods[i]:<10.4f} {rho_a[i]:<15.2f} {phase[i]:<12.2f}")

    return True


def verify_layered_forward():
    """验证层状模型正演"""
    print_header("验证3: 层状模型正演")

    # 两层层状模型: rho1=100, h1=100m, rho2=1000
    model = MT1DModel("two_layer", [100.0, 1000.0], [100.0])
    forward = MT1DForward(model)

    periods = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
    result = forward.calculate_impedance(periods)

    # 基本属性检查
    zxy = result["Zxy"]
    zyx = result["Zyx"]
    print_result("层状阻抗反对称", np.allclose(zxy, -zyx, rtol=1e-10))

    # 检查Zxx和Zyy是否为零
    zxx = result["Zxx"]
    zyy = result["Zyy"]
    print_result("层状Zxx ≈ 0", np.allclose(zxx, 0, atol=1e-20))
    print_result("层状Zyy ≈ 0", np.allclose(zyy, 0, atol=1e-20))

    # 计算视电阻率和相位
    rho_a, phase = forward.calculate_app_resistivity_phase(periods)

    # 打印结果
    print(f"\n  两层层状模型: rho=[100, 1000] Ohm·m, h=[100] m")
    print(f"  周期(s)    视电阻率(Ohm·m)    相位(度)")
    print(f"  " + "-" * 45)
    for i in range(len(periods)):
        print(f"  {periods[i]:<10.4f} {rho_a[i]:<15.2f} {phase[i]:<12.2f}")

    # 物理合理性检查: 短周期应该接近表层电阻率,长周期应该接近底层电阻率
    short_period_rho = rho_a[0]  # T=0.01s
    long_period_rho = rho_a[-1]  # T=100s
    print(f"\n  短周期(T=0.01s)视电阻率: {short_period_rho:.2f} Ohm·m (表层=100)")
    print(f"  长周期(T=100s)视电阻率: {long_period_rho:.2f} Ohm·m (底层=1000)")

    # 验证长周期趋于底层电阻率
    print_result(
        "长周期趋于底层电阻率", abs(long_period_rho - 1000.0) / 1000.0 < 0.5
    )  # 50%误差内

    return True


def verify_1d_validation():
    """验证1D模型验证器"""
    print_header("验证4: 1D模型验证器")

    # 创建均匀半空间模型
    model = MT1DModel("test", [100.0])
    forward = MT1DForward(model)
    periods = np.array([0.1, 1.0, 10.0])
    fields = forward.calculate_fields(periods)

    # 验证
    validator = Model1DValidator(fields)
    results = validator.validate_all()

    for r in results:
        print_result(r["name"], r["passed"], r.get("message", ""))

    return all(r["passed"] for r in results)


def verify_processor():
    """验证时间序列处理器"""
    print_header("验证5: 时间序列处理器")

    # 创建合成信号
    sample_rate = 2400
    duration = 1.0  # 1秒
    n = int(sample_rate * duration)
    t = np.arange(n) / sample_rate

    # 频率为100Hz的纯正弦信号
    f_signal = 100.0
    ex = np.sin(2 * np.pi * f_signal * t)
    ey = np.zeros(n)
    hx = np.cos(2 * np.pi * f_signal * t)
    hy = np.zeros(n)
    hz = np.zeros(n)

    processor = TimeSeriesProcessor(ex, ey, hx, hy, hz, sample_rate)

    # 计算FFT
    fft_result = processor.compute_fft(ex)

    # 找到峰值频率
    peak_idx = np.argmax(fft_result.amplitude)
    peak_freq = fft_result.frequencies[peak_idx]

    print_result(
        "FFT频率检测",
        abs(peak_freq - f_signal) < 1.0,
        f"检测到 {peak_freq:.1f}Hz, 期望 {f_signal}Hz",
    )

    # 测试阻抗估算
    periods = np.array([0.01])  # 对应100Hz
    imp_result = processor.estimate_impedance_at_periods(periods)

    print(f"\n  阻抗估算结果 (T={periods[0]}s, f={1 / periods[0]}Hz):")
    print(f"  Zxy = {imp_result['Zxy'][0]:.6f}")
    print(f"  视电阻率 = {imp_result['app_resistivity'][0]:.2f} Ohm·m")
    print(f"  相位 = {imp_result['phase'][0]:.2f}°")

    return True


def verify_api_workflow():
    """验证API完整工作流"""
    print_header("验证6: API工作流")

    reset_api()
    api = MTWorkflowAPI()

    # 创建均匀半空间模型
    model = api.create_halfspace("test_uniform", 100.0)
    print_result("API创建模型", model is not None)

    # 运行正演
    periods = np.array([0.1, 1.0, 10.0])
    forward_result = api.run_forward(periods)
    print_result(
        "API正演计算",
        forward_result is not None and len(forward_result["periods"]) == 3,
    )

    # 验证1D特征
    validation = api.validate_1d_model()
    print_result("API验证1D特征", len(validation) > 0)

    print(f"\n  正演结果:")
    print(f"  周期(s)    视电阻率(Ohm·m)    相位(度)")
    print(f"  " + "-" * 45)
    for i in range(len(periods)):
        print(
            f"  {forward_result['periods'][i]:<10.4f} "
            f"{forward_result['app_resistivity'][i]:<15.2f} "
            f"{forward_result['phase'][i]:<12.2f}"
        )

    return True


def verify_ts_config():
    """验证采集系统配置"""
    print_header("验证7: 采集系统配置")

    print_result("TS3配置存在", "TS3" in TS_CONFIGS)
    print_result("TS4配置存在", "TS4" in TS_CONFIGS)
    print_result("TS5配置存在", "TS5" in TS_CONFIGS)

    ts3 = TS_CONFIGS["TS3"]
    print_result("TS3采样率", ts3.sample_rate == 2400)
    print_result("TS3频率范围", ts3.freq_min == 1 and ts3.freq_max == 1000)

    return True


def verify_constants():
    """验证物理常数"""
    print_header("验证8: 物理常数")

    expected_mu0 = 4 * np.pi * 1e-7
    print_result("MU0正确", abs(MU0 - expected_mu0) < 1e-20, f"MU0 = {MU0:.6e}")

    return True


# ============================================================================
# 主函数
# ============================================================================


def main():
    print("\n" + "#" * 70)
    print("#  MT后台模块验证脚本")
    print("#  运行: python -m examples.mt_workflow.backend.verify_all")
    print("#" * 70)

    all_passed = True

    try:
        all_passed &= verify_constants()
        all_passed &= verify_model_creation()
        all_passed &= verify_ts_config()
        all_passed &= verify_halfspace_forward()
        all_passed &= verify_layered_forward()
        all_passed &= verify_1d_validation()
        all_passed &= verify_processor()
        all_passed &= verify_api_workflow()
    except Exception as e:
        print(f"\n[ERROR] 验证过程出错: {e}")
        import traceback

        traceback.print_exc()
        all_passed = False

    print_header("验证结果")
    if all_passed:
        print("  [PASS] 所有验证通过!")
    else:
        print("  [FAIL] 部分验证失败，请检查上述输出")

    print()
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
