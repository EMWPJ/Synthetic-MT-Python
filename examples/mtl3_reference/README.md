# mtl3 合成标定端到端测试夹具

H 型层状模型（100/10/1000 Ω·m，层厚 300/3000 m，底半空间）——
跨工具标定对拍的回归夹具（docs/parity_mtproc_phoenix_a0caf41.md）。

## 文件

| 文件 | 说明 |
|---|---|
| `mtl3_96.XMLSITE` | 97 锚点 0.0005–320 Hz（2^(1/5) 间隔），主夹具 |
| `mtl3.XMLSITE` | 47 锚点变体（8 点/倍频程） |
| `mtl3_96_phoenix.csv` | 24h 渲染 → mtproc 默认（phoenix）参考输出 |
| `mtl3_47_phoenix.csv` / `mtl3_47_cts.csv` | 47 锚点变体的两种标定参考 |

## 再生（确定性：同 seed 逐字节一致）

```bash
python -m synthts -model examples/mtl3_96.XMLSITE -out <out> \
    -scale 10 -seed 42 -dur 86400 -dense 0 -source noise -preset pure-l1
# 标定链随附：cp synthts/assets/{4360.CLB,BMT50000.CLC,BMT53014.CLC,BMT53025.CLC} <out>/
```

处理（mt-skills mt-process-plot）：

```bash
python mt-process-plot/scripts/mtproc.py proc <out> \
    --fmin 0.0005 --fmax 320 --per-decade 8 -o <out>/phoenix.csv   # 默认 phoenix
python mt-process-plot/scripts/mtproc.py proc <out> --cal cts ...  # cts 快照口径
```

真值公式（z1d 递归）见 `tools/plot_demo_amt.py`（RES=[100,10,1000]、
THICK=[300,3000]）。验收口径（2026-11-03 修复后）：默认 phoenix ρa
med ≤1.5%（四频段）、φ 恒定 180° 为写盘口径差；`--cal cts` 全频段
≤3.8%、无翻转。详见上方对拍文档 §7/§8。
