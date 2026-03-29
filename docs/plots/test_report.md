# MT 1D 正演-合成-处理-对比 测试报告

测试时间: 2026-03-29 23:01:31

## 测试配置

- 合成方法: 确定性合成 (DeterministicTimeSeriesSynthesizer)
- 频段: TS3 (2400Hz, 1-1000Hz)
- 合成时长: 60秒
- 误差容限: 视电阻率<2.0%, 相位<1.0°

## 测试结果汇总

| 模型 | 最大rho_a误差% | 平均rho_a误差% | 最大相位误差° | 平均相位误差° | 状态 |
|------|---------------|---------------|--------------|---------------|------|
| uniform_100 | 0.9483 | 0.3044 | 0.3881 | 0.0973 | PASS |
| uniform_1000 | 0.9483 | 0.3044 | 0.3881 | 0.0973 | PASS |
| two_layer_hl | 1.1613 | 0.3603 | 0.4791 | 0.1178 | PASS |
| two_layer_ll | 0.3387 | 0.1281 | 0.1149 | 0.0345 | PASS |
| three_layer_hll | 0.6242 | 0.2479 | 0.2052 | 0.0626 | PASS |

## 生成图像

- 单模型对比图: `docs/plots/<model_name>_comparison.png`
- 汇总对比图: `docs/plots/all_models_summary.png`
- 误差汇总图: `docs/plots/error_summary.png`
