# MT 1D 正演-合成-处理-对比 测试报告

测试时间: 2026-03-30 11:41:30

## 测试配置

- 合成方法: 确定性合成 (DeterministicTimeSeriesSynthesizer)
- 频段: TS3 (2400Hz, 1-1000Hz)
- 合成时长: 60秒
- 误差容限: 视电阻率<2.0%, 相位<1.0°

## 测试结果汇总

| 模型 | 最大rho_a误差% | 平均rho_a误差% | 最大相位误差° | 平均相位误差° | 状态 |
|------|---------------|---------------|--------------|---------------|------|
| uniform_100 | 0.8912 | 0.2248 | 0.4310 | 0.0788 | PASS |
| uniform_1000 | 0.8912 | 0.2248 | 0.4310 | 0.0788 | PASS |
| two_layer_hl | 0.8778 | 0.2750 | 0.4965 | 0.1224 | PASS |
| two_layer_ll | 0.4016 | 0.1102 | 0.2370 | 0.0393 | PASS |
| three_layer_hll | 0.6654 | 0.1942 | 0.3148 | 0.0767 | PASS |

## 生成图像

- 单模型对比图: `docs/plots/<model_name>_comparison.png`
- 汇总对比图: `docs/plots/all_models_summary.png`
- 误差汇总图: `docs/plots/error_summary.png`
