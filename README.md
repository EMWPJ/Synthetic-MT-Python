# Synthetic MT TimeSeries (SMTTS) — Python · synthts

[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-green.svg)](https://www.python.org/)
[![numpy only](https://img.shields.io/badge/dependency-numpy%20only-orange.svg)](https://numpy.org/)

**English** | [中文](#中文)

## What this is

`synthts` turns **XMLSITE forward-model files** into calibrated **Phoenix TS2–TS5 + TBL
binary time series**, and ships a minimal MT processor (`mtprocess`) that verifies the
synthetic data against the model's theoretical impedance. It inherits the name and
method of the Delphi-based original, [Synthetic MT TimeSeries (SMTTS)](https://github.com/EMWPJ/SyntheticMTTimeSeries),
and continues it as a lean CLI + Python library; the synthesis method is that of
Wang et al. (2023), cited in full below. Version 1.9 is a lean,
scriptable, AI-agent-friendly line of this repository: one algorithm, one pipeline,
numpy as the only runtime dependency.

```
XMLSITE model(s) → built-in calibration → coherently-paired synthesis
                  → Phoenix TS2–TS5 + TBL on disk → mtprocess verification
```

Method reference:

> Wang P, Chen X, Zhang Y (2023). [Synthesizing magnetotelluric time series based on
> forward modeling](https://www.frontiersin.org/articles/10.3389/feart.2023.1086749/full).
> *Frontiers in Earth Science* 11:1086749.

### Highlights

- **Two source engines** (v2.0, `-source`, default `noise`):
  - **noise** — a three-layer natural source: continuous white-noise floor
    colored by the forward mapping (true continuous spectrum by construction),
    half-octave band-burst log-normal gain envelopes (a band signal that lasts
    a while — bursty, polarization-wandering, statistically realistic), and an
    optional sparse Gabor event layer (`-preset full-event`).
  - **lines** — the v1 line-spectrum engine (`RandomSegmentPartiallyWindowed`,
    Wang et al. 2023), kept byte-identical as a deterministic, grid-matched
    reference (also the fitting basis for source-parameter estimation work).
- **Built-in calibration** — two embedded CTS tables for a reference instrument (Phoenix
  box 4360 + BMT coils), one set per mode. No external DLLs, no calibration files to find.
- **Coherently-paired multi-site batches** — pass several models via `-model` and the
  source process is shared bin-exactly across all sites: one source field exciting
  many sites (remote-reference scenarios).
- **Presets & parameter injection** — `-preset natural` (default) / `pure-l1` /
  `full-event`, or override any source parameter via `-source-params FILE.json`.
- **Deterministic** — `-seed` (default 42) fixes every random draw; the same command
  produces byte-identical output.
- **Machine-readable completion summary** — a final `SYNTHTS_SUMMARY {json}` line on
  stdout for scripts and AI agents.
- **Self-verifying** — `tools/mt_e2e_demo.py --source both` checks four quality
  gates per engine; `tools/par_report.py` produces a lines-vs-noise comparison
  report; `tools/cost_bench.py` benchmarks both engines.

### Removed in v1.9

This line deliberately drops the former kitchen-sink feature set: the six synthesis
algorithms (only `RANDOM_SEG_PARTIAL` survives), the PySide6 GUIs, noise injection,
ModEM import, GMT/CSV/NumPy exporters and the 1D workflow shell. They remain available
under the [`v1.8-legacy`](https://github.com/EMWPJ/Synthetic-MT-Python/tree/v1.8-legacy)
tag — the last pre-rewrite state — and the git history.

## Installation

Python 3.10+; numpy is the only runtime dependency.

```bash
git clone https://github.com/EMWPJ/Synthetic-MT-Python.git
cd Synthetic-MT-Python
pip install -e .
```

If the `synthts` command is not on your `PATH` afterwards (a user-site Scripts directory
is common on Windows), use `python -m synthts` instead — both entry points are equivalent.

The paper-style figures additionally need matplotlib: `pip install -e ".[plot]"`.

## Quick start

```bash
# 1) Single site — MT mode with defaults (72 h duration, seed 42, dense 4/octave)
synthts -model examples/mta.XMLSITE -out out_mta

# 2) Multi-site coherently-paired batch (shared random source per frequency)
synthts -model examples/mta.XMLSITE examples/mtb.XMLSITE -out out_pair -dur 3600

# 3) End-to-end verification — all four gates must print PASS
python tools/mt_e2e_demo.py
```

`python -m synthts ...` is equivalent to the `synthts` console command.

## CLI reference

| Option | Default | Description |
|---|---|---|
| `-model FILE [FILE ...]` | required | One or more `.XMLSITE` models. Several files = coherently-paired multi-site batch. |
| `-mode MT\|AMT` | `MT` | `MT` writes TS3/TS4/TS5; `AMT` writes TS2/TS3/TS4. |
| `-out DIR` | required | Output directory (created if missing). |
| `-dur SECONDS` | MT: 259200 (72 h); AMT: 3600 | Synthesis duration. |
| `-periods N` | `4` | Mean segment length in signal periods. Dense grids: ≥ 128 recommended (the e2e demo uses 128). |
| `-scale X` | `1.0` | Field amplitude factor. |
| `-rot EX,EY,HX,HY` | `0,0,0,0` | Channel rotations in degrees. |
| `-seed N` | `42` | numpy Generator seed. Same seed → byte-identical output. |
| `-dense N` | `4` | Log-domain densification, points per octave (`0` disables). |
| `-log FILE` | off | Tee the human-readable log to a file. |

Fixed behaviour:

- Start time is pinned to **2050-01-01 00:00:00**; each output TBL's `FTIM`/`LTIM`
  are rewritten to the actual synthesized span.
- Exit codes: **0** success, **2** failure (error message on stdout; set the
  environment variable `SYNTHTS_DEBUG=1` for a full traceback).
- Duplicate site names are rejected (outputs would collide).

## Machine-readable output

The last line printed to **stdout** is a single-line JSON summary (it is never written
to the `-log` file):

```
SYNTHTS_SUMMARY {"tool": "synthts", "version": "1.9.0", "mode": "MT", "models": [...],
 "sites": ["MTA"], "n_freqs": [51], "schemas": ["TS3", "TS4", "TS5"],
 "begin": "2050-01-01 00:00:00", "end": "2050-01-04 00:00:00", "dur_s": 259200.0,
 "periods": 4.0, "scale": 1.0, "rot": "0,0,0,0", "seed": 42, "dense": 4.0,
 "out_dir": "...", "files": ["...MTA.TBL", "...MTA.TS3", ...], "n_files": 5,
 "elapsed_s": 3.812}
```

| Field | Meaning |
|---|---|
| `mode`, `models`, `sites` | Mode, input model paths as given, resolved site names. |
| `n_freqs` | Effective (post-densification) signal frequency count per site. |
| `schemas` | TS bands written. |
| `begin`, `end`, `dur_s` | Actual synthesized time span (fixed epoch + duration). |
| `periods`, `scale`, `rot`, `seed`, `dense` | Effective synthesis parameters. |
| `files`, `n_files` | Every file written this run (TBL, TS bands, dense XMLSITE). |
| `elapsed_s` | Wall-clock seconds. |

## Python API

```python
from synthts.cli import main as synth_main
from synthts import mtprocess

rc = synth_main(['-model', 'examples/mta.XMLSITE', '-out', 'out_mta',
                 '-dur', '3600', '-periods', '64', '-dense', '0'])
rows = mtprocess.process('examples/mta.XMLSITE', 'out_mta')
# rows: (freq, Z_est 2x2 complex, Z_theory 2x2, n_windows, max_relative_error)
```

Full runnable script: [`examples/example_mtprocess.py`](examples/example_mtprocess.py).

## Output files

Per site in `-out`:

| File | Content |
|---|---|
| `<site>.TBL` | Phoenix table file (FTIM/LTIM set to the actual span). |
| `<site>.TS2` … `<site>.TS5` | Phoenix time series: 32-byte record headers + 24-bit little-endian, channel-interleaved samples. |
| `<site>_dense.XMLSITE` | The densified model actually synthesized (present when `-dense > 0`); can be fed straight to `mtprocess`. |

## Verification gates (`tools/mt_e2e_demo.py`)

The demo synthesizes two sites (half-space ρ=100 Ω·m; ρ=30 Ω·m with perturbed impedance
and 35° rotated geometry), then checks:

1. **Anchor impedance** — single-frequency windowed DFT + calibration + least-squares Z
   vs theory (sparse grid < 1e-2; densified grid < 1e-1, limited by neighbouring-segment
   scattering, a resolution limit not an error).
2. **Inter-site coherence** — H of site B = M·H of site A with a constant 2×2 M
   (relative residual < 2e-2; measured < 2e-3).
3. **Full dense-grid impedance** — median < 1e-2, ≥ 90 % of frequencies < 5e-2.
4. **PSD continuity** — in-gap spectral amplitude ratio improved ≥ 46× by densification.

The full demo passes on seed 42; the marginal dense-grid gate was additionally
verified on seeds 7 and 123.

## Paper-style figures

```bash
pip install -e ".[plot]"
python tools/plot_demo.py        # writes ./plot_out/fig_time_series.png + fig_rho_phase.png
```

Draws two figures in the style of Wang et al. (2023) for a three-layer demo model
(100/10/1000 Ω·m, H-type curve):

- `fig_time_series.png` — channel x samplerate panel grid (rows Ex/Ey/Hx/Hy,
  columns 2400/150/15 Hz; a 1D model has no vertical magnetic field, so Hz is
  not drawn) plus a dual-axis PSD panel (E fields left, H fields right, one
  curve family per samplerate) — paper Fig 5 layout;
- `fig_rho_phase.png` — apparent resistivity and phase estimated from the
  synthetic series by least squares vs the 1D forward theory, on a reversed
  log frequency axis (high frequency left): theoretical black lines, synthetic
  XY as red circles and YX as blue triangles — paper Fig 6 style.

## Performance (reference: MT mode, NVMe)

| Job | lines (dense=4, periods=64) | noise (natural preset) |
|---|---|---|
| 1 h × 2 sites | 0.17 s | 0.54 s |
| 1 day × 1 site | 4.4 s | — |
| 10 days × 1 site | 54 s | — |
| 10 days × 10 sites (coherently paired) | 127 s | — |
| 72 h × 1 site | — | ~121 s (TS5 band-synthesis FFTs dominate) |

lines numbers are the v1 engine (sparse comb matched to the processing window);
noise runs are cost-decoupled from the model frequency count (anchor-count
sensitivity <10%, interleaved benchmark; see `tools/cost_bench.py`).

## Repository layout

```
Synthetic-MT-Python/
├── synthts/            # the package (CLI, synthesis, calibration, I/O, mtprocess)
│   └── assets/         # built-in TBL/CLB/CLC instrument sets (MT + AMT)
├── examples/           # ready-to-run XMLSITE models + Python API example
├── tools/
│   ├── mt_e2e_demo.py  # four-gate end-to-end verification
│   └── plot_demo.py    # paper-style figures (matplotlib extra)
├── tests/              # pytest: smoke (MT/AMT/determinism) + slow four-gate e2e
├── SKILL.md            # playbook for AI agents calling the CLI
├── pyproject.toml
└── LICENSE             # GPL-3.0
```

## Tests

```bash
pytest                # smoke: MT + AMT CLI runs, JSON summary, same-seed determinism
pytest -m slow        # four-gate end-to-end demo + paper-style figures
```

## Citation

```bibtex
@article{wang2023synthesizing,
  title={Synthesizing magnetotelluric time series based on forward modeling},
  author={Wang, Peijie and Chen, Xuan and Zhang, Yong},
  journal={Frontiers in Earth Science},
  volume={11},
  pages={1086749},
  year={2023},
  publisher={Frontiers}
}
```

## Related projects

- [SyntheticMTTimeSeries](https://github.com/EMWPJ/SyntheticMTTimeSeries) — the original Delphi implementation (this project inherits its name)
- [Modified-ModEM](https://github.com/EMWPJ/Modified-ModEM-with-output-electromagnetic-field-values) — ModEM with EM field output (source of forward-model files)

## License

[GPL-3.0](LICENSE)

---

## 中文

# Synthetic MT TimeSeries（SMTTS）Python 版 — Phoenix 格式 MT/AMT 合成时间序列

项目名继承自 Delphi 原版 [SyntheticMTTimeSeries](https://github.com/EMWPJ/SyntheticMTTimeSeries)。

把 **XMLSITE 正演模型**合成为带内置标定的 **Phoenix TS2–TS5 + TBL 二进制时间序列**，
并附带极简 MT 处理器 `mtprocess` 验证合成数据与理论阻抗一致。**v2.0 起默认双引擎**
（`-source`，默认 `noise`）：`noise`=三层天然源（连续谱底+分带突发包络+可选事件层，
谱连续、阵发、统计真实），`lines`=v1 线谱（RandomSegmentPartiallyWindowed，Wang et al.
2023，逐字节保留，确定性网格参照/源参数拟合基）；同源批量多测点（远参考场景）、
预设 natural/pure-l1/full-event + `-source-params` 参数注入、`-seed` 全程可复现、
结束时输出机器可读的 `SYNTHTS_SUMMARY` JSON 摘要行。

**安装**（Python 3.10+）：`pip install -e .`

**用法**：

```bash
synthts -model examples/mta.XMLSITE -out out_mta                          # 单测点（MT 模式缺省 72h，noise 引擎）
synthts -model examples/mta.XMLSITE examples/mtb.XMLSITE -out out_pair -dur 3600   # 多测点同源批量
synthts -model examples/mta.XMLSITE -out out_ref -source lines            # v1 线谱引擎
python tools/mt_e2e_demo.py --source both                                # 双引擎四闸门验证
```

机器可读摘要（stdout 末行，不写入 `-log`）：
`SYNTHTS_SUMMARY {"mode": "MT", "sites": ["MTA"], "n_files": 5, "files": [...]}`。

时间起点固定 2050-01-01 00:00:00；`-dur` 缺省 AMT=3600s、MT=72h；退出码 0 成功 / 2 失败。
完整参数表与 JSON 摘要字段见英文部分。v1.9 删除了旧版的六算法、PySide6 GUI、噪声注入、
ModEM 导入与 GMT/CSV 输出（旧状态钉在 [`v1.8-legacy`](https://github.com/EMWPJ/Synthetic-MT-Python/tree/v1.8-legacy) 标签）。

许可证：GPL-3.0。
