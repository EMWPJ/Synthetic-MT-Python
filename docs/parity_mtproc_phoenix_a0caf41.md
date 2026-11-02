# 对拍证据：SMOOTH 渲染 ↔ mtproc phoenix 在 0.3–320 Hz 不一致（a0caf41，2026-11-03）

> 提交者：macOS 侧端到端验证。结论先行：**a0caf41 起合成端 SMOOTH 渲染与
> mtproc 默认 phoenix（syscal_native 求值）在 0.347–320 Hz 互相不一致**
> （ρa 误差 ±10%…−93%/+859%，φ 翻 180°），≤0.26 Hz 双侧一致到 0.06%。
> synthts 门禁（test_gates_v2 exit 0）抓不到——门禁的渲染/处理同为稠密
> 插值，自洽；跨工具配对只有"synthts 渲染 → mtproc 处理"能暴露。

## 1. 现象（24 h 层状合成 100/10/1000 Ω·m，96 密锚点 0.0005–320 Hz，seed 42）

同一模型、同一 seed，两种链路的 mtproc 处理结果 vs 精确 z1d 理论：

| 频段 | 旧链（nearest 渲染 + `--cal cts` 整段预标定，534bba5） | 新链（SMOOTH 渲染 + mtproc 默认 phoenix） |
|---|---|---|
| 0.0005–0.26 Hz | rms 1.86% | 与旧链一致（±0.06% 内）✓ |
| 0.347–320 Hz | ±1.4% | **+46.7%、−93.5%、+859%、φ 翻 180°** ✗ |

## 2. 复现

```bash
# 合成（SMOOTH 现为默认）
cd synthts && python -m synthts -model mtl3.XMLSITE -out data_v2 \
    -scale 10 -seed 42 -dur 86400 -dense 0 -source noise -preset pure-l1
# mtproc 默认标定（a0caf41 起 = phoenix）
python mt-process-plot/scripts/mtproc.py proc data_v2 --site MTL \
    --fmin 0.0005 --fmax 320 --per-decade 8 -o out.csv
# 对照组（旧链）：SMOOTH_RESPONDS=False 渲染 + --cal cts
```
TBL 需含 HXSN/HYSN/HZSN 串号（synthts DEFAULT.TBL 为精简版需注入；
mtproc 串号正则已支持 BMT 族，0d4bcaf）。完整夹具与逐频数据：
macOS `/Users/wpj/code/mt-skills-demo/proc/mtl3_v2.csv`（新链）与
`proc/mtl3_fixed.csv`（旧链）。

## 3. 对拍：稠密表插值 vs syscal_native 直接求值（同一 DEFAULT 仪器链）

对拍判据：respond 形式（adcFactor·conj(z)/|z|²）与 native 求值 z 的
一致性 = |respond·conj(z_native)| ≡ adcFactor。

| band | 幅值相对差 median | 幅值相对差 max | 相位差 median | 相位差 max |
|---|---|---|---|---|
| 5（0.3–3.75 Hz） | 4.6e-3 | 1.34e-1 | **89.35°** | 167.46° |
| 4（3.75–72 Hz） | 9.3e-3 | **1.47e+2**（7.493 Hz，表下限钳位区） | 88.08° | 179.71° |
| 3（75–320 Hz） | 4.4e-2 | 1.88e-1 | 68.53° | 170.42° |

- **相位中位 89°**（band5）≈ 差一个 iω 因子——疑线圈低频外推（iω 模型）
  在稠密表生成与 syscal_native 求值两侧的约定/实现不同。
- band4 下限钳位区差 147×——稠密表带下限（7.5 Hz）外的钳位行为两侧不同。
- 佐证：syscal_native vs cts_snap.json（SysCal_V7.exe 金标，60 锚点 ×
  5 通道）本机实测 **max 相对差 4.6e-4** → native 侧贴合 exe 金标；
  偏离主要在稠密表/其插值一侧（另实测稠密表在快照锚点上幅值呈系统性
  2× 平移，成因未定，供排查线索）。

## 4. 门禁盲区

test_gates_v2 的处理走 synthts.mtprocess（同用稠密插值），渲染/处理
自洽 → SMOOTH 切换对其不可见。建议增补跨工具门：synthts 渲染 →
mtproc `--cal phoenix` 端到端 ρa/φ 对拍（本文件 §1 夹具即可）。

## 5. 影响与现状

- a0caf41 起，mtproc **默认标定（phoenix）处理新渲染数据在 0.3–320 Hz
  不可信**（±10%~−93%/+859%，φ 翻转）；≤0.26 Hz 正常。
- 旧渲染数据（2026-11-02 前）用 `--cal cts` 仍正确（阶梯自洽，
  mtproc 已含整段频域预标定修复 534bba5）。
- macOS 侧未 revert（遵裁定保持现状），等稠密表/口径对齐修复后
  重新对拍本文件 §3 判据（应趋 0）。

## 6. 验证环境

macOS arm64 · Python 3.13.12 · numpy 2.4.6 · 提交 a0caf41 ·
mt-calibration syscal_native（净室移植，Windows 对拍位级一致）。

---

## 7. 结案（2026-11-03，Windows 侧修复后复验）

根因两条，均已修（上游 master 与 mt-skills 同步）：

1. **band 路由错**：渲染按频率值（band_of_freq）烤响应，处理端按文件
   采样率去标定——TS5 的 0.8–7.5 Hz bins 烤了 band4 响应（7 Hz 差 70×），
   TS4 的 8–75 Hz 同理。修复：渲染/处理孪生/物理域闸门全部改为按文件
   采样率路由（`calib.band_of_sr`，`compute_responds`/`responds_on_grid`
   增 `band=` 参数；None 仅存留 legacy 旧渲染配对）。
2. **稠密表覆盖不足 + T 空间符号歧义**：旧表 band5 止于 0.773 Hz、
   band4 止于 7.79 Hz（本文件 §3 的"表下限钳位 147×"即此）；且 T 在
   z 零/极点处符号歧义（§3 的"相位中位 89°"含钳位与卷绕两者）。修复：
   `tools/gen_dense_cts.py` 重写——syscal_native 直接求值（废除旧版逐
   band 对齐 hack，那正是"稠密表在快照锚点系统性平移"的来源），每 band
   全计划覆盖（band2 9–11000 / band3 0.9–1250 / band4 0.009–80 /
   band5 1.4e-5–7.7 Hz），48 点/倍频程 + 自适应加密（相位步 >10° 或幅值
   变 >5% 二分，8 轮），存原始 z、运行时 z 空间插值后 transform。

复验（Windows，同引擎链）：本文件 §3 判据 dense ≡ native——
幅值差 median 2e-5…1.5e-4、max ≤1.7%；相位差 max ≤0.1°（全 band 全频段
200 点/带扫频）。跨工具端到端：新渲染 → mtproc（cts，band 感知）vs
synthts estimate_impedance_at——复数 Z 比值 0.96–1.01、相位差 ≤1.7°。
遗留边界（插值极限，非缺陷）：50 Hz 陷波中心 ±0.15 Hz（dll 原始退化行
已滤，跨缺口线性内插）、各档 Nyquist 缘 ~3%（响应→0）、band2 >11.3 kHz
（标定记录括号极限，钳位）。synthts 数据经 mtproc 处理请用 `--cal cts`
（写盘极性/μ0 口径与渲染端成对）；`--cal phoenix` 面向真 Phoenix 数据
（E 符号口径不同，合成数据会差 180°）。

## 7. 复测（2026-11-03，b12013f/be33dc9 拉取后）：未修复，且渲染侧稠密表相位对不上 exe 金标

b12013f「合成标定按文件采样率路由 + 稠密表全计划覆盖重生成（修 Mac 端
跨工具回归 de40aa6）」拉取后复测（同 §1 夹具，SMOOTH 渲染 + mtproc
默认 phoenix，vs 精确 z1d）：

| 频段 | ρa median | ρa max |
|---|---|---|
| 0.0005–0.002 Hz | −1.29% | 3.06% ✓ |
| 0.002–0.05 Hz | −3.55% | 4.85% ✓ |
| 0.05–0.2 Hz | −0.44% | 0.83% ✓ |
| **0.4–3.75 Hz** | **−54.08%** | **87.48%** ✗ |
| **3.75–72 Hz** | **+8.24%** | **146.24%** ✗ |
| **72–320 Hz** | **+31.50%** | **377.89%** ✗ |

φ |err| median **179.23°**（整体符号翻转）。归属判定（新增）：
**重生成的 cts_dense 在 band4/band5 相位上偏离 exe 金标**——
calib.responds_on_grid（渲染侧实际使用的表）vs cts_snap（SysCal_V7.exe
金标）逐锚点对拍：band4 相位差中位 **99.27°**/max 175.86°（6 锚点几乎
全体翻转）、band5 max 158.40°（f=6 Hz ch=Hz）；band2/band3 完美
（幅值 4e-8~5e-5、相位 0.00°）。疑点：0fe8d3f 的 acd 支路语义修正
（不再忠实复刻 exe）被带进 gen_dense_cts.py 重生成，与未修正语义的
syscal_native/exe 金标分叉（band4 相位中位 99° ≈ ±(180−80)° 混合）。
interim 处理组合：旧渲染数据 + `--cal cts` 全频段正确（本机复验
median −0.94%、rms 2.22%、φ 0.16°）；SMOOTH 渲染数据在两侧对齐前
不要经 mtproc 处理。

---

## 8. 第二轮结案（2026-11-03，Windows 复现 → 定位 → 修复）

对 b12013f 复测结论的核对与推进（Windows 同夹具复现：φ 全频段 ~180°、
ρa 0.3 Hz 以上 −35/−65/+19%，与 Mac 端一致——**可复现，非 Mac 环境问题**）：

- **§7 的 dense ≡ native 结论经复核成立**（同引擎同表逐点对拍 300 点/band
  全计划范围：幅值 med 2e-5、max ≤1.7%、相位 ≤0.1°）。"band4/5 相位偏离
  快照 99°/158°" 的归属判定需修正：exe 稀疏快照在锚点间存在 **z→−z 约定
  翻转**（该盒 CLB 真值符号交替，AGENTS 既有结论）——快照的逐锚点相位
  不是合法相位基准（band2/3 恰无翻转所以"完美"，band4/5 有翻转所以
  "99°/158°"），与 acd 语义修正无关。
- **端到端崩坏的真正根因（第三条，独立于路由/表）**：mtproc phoenix 段
  路径对**全 Nyquist bin 网格**逐 bin 除以 raw native z——z 在记录括号缘
  （band5 实测 ~7.35 Hz 起 541 bin）与陷波中心趋 0，C=1e-9/z 产生巨型乘子
  （既有坏 bin 守卫只拦 inf/|C|>1e30，tiny-nonzero z 的 C~1e9+ 被保留），
  乘进谱再 irfft → 单 bin 能量扩散成整段振铃，0.35 Hz 以上短窗全污染。
  cts 乘的 T_snap 只取 exe 锚点有限值，天然免疫；synthts 引擎直读 counts
  同样不经过此路径——这就是三路对拍中唯 phoenix 崩的原因。
- **修复**：mt_calib.calibrate_segment phoenix 分支除法前自适应退化 bin
  零化（|z| < 1e-3×该通道中位，−60 dB；响应物理上不可能逐 bin 跌 60 dB）。
  复验（同夹具同 seed）：phoenix 默认 vs z1d——ρa med −0.69%/+0.69%/
  +0.97%/−2.25%（四频段），max 13.6%（仅 72–320 边缘）。
- **遗留：φ 恒定 180°（E 符号口径，非缺陷非回归）**：cts 的 C[:2] 带
  负号（与 synthts 写盘负号成对，z1d 复验 φ ≤2.5°），phoenix 除法无负号
  （对齐 Delphi RemoveSystemRespond 语义，面向真 Phoenix 数据）。两种口径
  差一个整体符号——**synthts 数据请用 `--cal cts`**（ρa/φ 全对）；真
  Phoenix 数据用 phoenix（极性以实测为准）。若要消除该符号差需统一写盘
  或处理端约定，涉及真实数据极性证据，留待裁定。
- 修复落点：mt-process-plot/scripts/mt_calib.py（phoenix 退化 bin 零化）；
  synthts 侧本轮无改动（路由/表修复经复验成立）。
