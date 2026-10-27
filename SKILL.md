---
name: synthts
description: Synthesize Phoenix-format MT/AMT time series from XMLSITE forward models. Use when a task mentions synthetic magnetotelluric data, generating TS2-TS5/TBL files, MT/AMT time-series simulation or benchmark data for MT processing pipelines, including coherently-paired multi-site (remote-reference) batches. CLI tool, numpy-only dependency, deterministic via seed, JSON completion summary. (合成大地电磁时间序列)
---

# synthts — AI agent playbook

Project: **Synthetic MT TimeSeries (SMTTS), Python edition** — inherits the name of
the Delphi original; the tool/package is `synthts`.
XMLSITE forward-model file(s) in → calibrated Phoenix TS2–TS5 + TBL binary out.
Two source engines since v2.0 (`-source`, default `noise`):
- `noise` — three-layer natural source (L1 continuous white-noise floor colored
  by the mapping; L2 half-octave band-burst log-normal gain envelopes with
  optional cross-band gamma; L3 sparse Gabor event layer). Continuous spectrum
  by construction, bursty, statistically realistic.
- `lines` — v1 line-spectrum engine (Wang et al. 2023), kept byte-identical.
Single pipeline, Python 3.10+, numpy only. Also a Python library
(`synthts.cli.main`, `synthts.mtprocess`).

Repository: https://github.com/EMWPJ/Synthetic-MT-Python (GPL-3.0).
Method paper: Wang P, Chen X, Zhang Y (2023), Frontiers in Earth Science 11:1086749,
https://www.frontiersin.org/articles/10.3389/feart.2023.1086749

## Install (once, in the repo root)

```bash
pip install -e .        # adds the `synthts` console command; `python -m synthts` also works
```

## CLI contract (stable)

```bash
synthts -model <one-or-more.XMLSITE> -out DIR [-mode MT|AMT] [-dur SECONDS]
        [-scale 1] [-seed 42] [-dense 4] [-log FILE]
        [-source noise|lines] [-preset natural|pure-l1|full-event]
        [-source-params FILE.json]

# ModEM 一键：三维模型 + 常规 dat → 补场值段 → 改版 exe 正演 → 测点 → 时序
synthts -modem-run MODEL DAT -out DIR -modem-exe EXE
        [-stations A,B] [-max-periods N] [-keep-work] [-modem-par PAR]
        [-mode MT] [-dur S] [-seed 42] ...        # 合成参数透传

# 其他模式（与 -model 互斥）
synthts -modem RESPONSE.dat -out DIR [-modem-units si|raw]   # 响应→测点
synthts -modem-add-fields DAT -out NEW_DAT                   # 补占位段
synthts -edi SITE.EDI -out DIR                               # 实测 EDI→测点（⚠ 非标准）
synthts -qc DIR                                              # 逐站 ρa/φ QC 图
synthts -info                                                # 频段/体积速查
synthts -selftest                                            # 安装自检（双引擎）
```

- Exit codes: **0** success, **2** failure (message on stdout; `SYNTHTS_DEBUG=1` shows the traceback).
- Start time is fixed at 2050-01-01 00:00:00; TBL `FTIM`/`LTIM` carry the actual span.
- `-dur` omitted → AMT 3600 s, MT 259200 s (72 h).
- v2.0+: **default source is `noise`** (preset `natural`); `-periods`/`-rot` are inert
  for noise and print a WARNING (use `-source lines` to exercise them).
- Presets: `natural` (L1 floor at −20 dB + L2 bursts, default) / `pure-l1`
  (continuous floor only — used by gates 5/6) / `full-event` (adds L3 events).
  Override any source parameter via `-source-params` JSON (schema:
  `synthts/source_params.py`; excludes seed/epoch).
- Machine-readable result — the **last stdout line** (never written to `-log`):

```
SYNTHTS_SUMMARY {"tool":"synthts","version":"...","mode":"MT","models":[...],"sites":[...],
"n_freqs":[...],"schemas":["TS3","TS4","TS5"],"begin":"2050-01-01 00:00:00","end":"...",
"dur_s":...,"periods":...,"scale":...,"rot":"0,0,0,0","seed":42,"dense":4.0,
"out_dir":"...","files":[...],"n_files":...,"elapsed_s":...}
```

Parse with: `line = [l for l in stdout.splitlines() if l.startswith('SYNTHTS_SUMMARY ')][-1]`,
then `json.loads(line[len('SYNTHTS_SUMMARY '):])`.

## Typical tasks

### 1. Single site, defaults (MT, 72 h, seed 42, dense 4/octave)

```bash
synthts -model examples/mta.XMLSITE -out out_mta
```

Produces `out_mta/MTA.TBL`, `MTA.TS3/.TS4/.TS5`, `MTA_dense.XMLSITE`. Runtime ≈ 54 s
for 10 days; MT output ≈ 250 MB/day/site — prefer explicit small `-dur` when testing.

### 2. Multi-site coherently-paired batch (remote-reference scenario)

```bash
synthts -model examples/mta.XMLSITE examples/mtb.XMLSITE -out out_pair -dur 3600
```

Shared frequencies get identical segment lengths and random amplitude/phase factors
across sites (H_B = M·H_A). Duplicate site names are rejected (exit 2).

### 3. Verify an install end to end (four gates, dual engine)

```bash
python tools/mt_e2e_demo.py --source both   # lines + noise engines, four gates each
```

Checks anchor impedances, inter-site coherence, full dense-grid impedance and
spectral continuity (lines: PSD-gap ratio; noise: slope-normalized continuity
probes >0.8); exit 0 only if both engines pass all four gates. Noise-mode Z
floors (median ~2e-2) reflect CTS nearest-neighbour steps × continuous spectra —
a processing physics floor, not an engine defect (design §5.3).

### 4. Estimate impedance from synthesized data (Python)

`examples/example_mtprocess.py` — `synthts.cli.main(...)` then
`mtprocess.process(model, out_dir)` → per-frequency `(freq, Z_est, Z_theory, nwin, max_rel_err)`.
For noise-source data pass `bypass_screen=True` (half-window screening is
designed for carrier phase jumps and passes only ~4% of noise windows).

### 5. Paper-style figures (Wang et al. 2023, Fig 5/6 style)

```bash
pip install -e ".[plot]"     # once, adds matplotlib
python tools/plot_demo.py    # writes ./plot_out/fig_time_series.png + fig_rho_phase.png
```

Layered 1D demo model; channel x samplerate panel grid (Ex/Ey/Hx/Hy at
2400/150/15 Hz) with a dual-axis PSD panel, and apparent resistivity/phase
(LSQ) vs theory on a reversed log frequency axis (theory = black lines;
synthetic XY = red circles, YX = blue triangles).

### 6. Dual-engine comparison report (v2)

```bash
python tools/par_report.py    # lines vs noise: Z / coherence / spectral / stats table, PASS/FAIL
```

### 7. ModEM 3-D model → synthetic time series in ONE command (推荐)

```bash
synthts -modem-run initialmodelfile datafile -out out1 \
        -modem-exe ModEM.exe -stations JH01,JH02 -max-periods 8 -dur 3600
# 产出：out1/<站>.XMLSITE + <站>_dense.XMLSITE + TS/TBL 时序 + fwd/ 日志
#       + modem_run_meta.json；末行 SYNTHTS_SUMMARY 为透传合成的摘要
```

One command runs: subset (`-stations` codes / `-max-periods` log-spaced,
forward cost ≈ 1 min/period on a COMMEMI-scale grid) → EM_Fields
placeholder append → **auto copy to a short-ASCII work dir** (the modified
exe's command line cannot hold long Chinese paths — arguments truncate,
run dies with "No input model parametrization"; override the work root
with `SYNTHTS_MODEM_WORK`) → `exe -F model p.dat r.dat f.E par` → parse
the filled response → XMLSITE sites → synthesis with your `-mode/-dur/
-seed/...` passthrough. `fwd/modem_fwd.log` + `fwd/response.dat` are kept;
the huge grid solution `f.E` is deleted unless `-keep-work`. Needs the
**modified** exe (EMWPJ repo) that fills EM_Fields; `-modem-par` optional
(default = tight tolerances).

Advanced (manual steps, e.g. cluster forward):

```bash
synthts -modem-add-fields datafile.dat -out datafile_fields.dat
ModEM.exe -F model datafile_fields.dat response.dat fields.E modempar
#  ⚠ 长中文路径须先拷到短 ASCII 路径（exe 命令行缓冲会截断）
synthts -modem response.dat -out sites/
synthts -model sites/*.XMLSITE -out ts/ -dur 3600
```

`-modem-add-fields` appends a `> EM_Fields` section to an ordinary ModEM
dat: for every (period, station) record in the source's own enumeration
order, ten rows `EPolEx..EPolHz BPolEx..BPolHz` initialized to 0 with
error `2.0E+15`; the source bytes are preserved verbatim (period/
coordinate tokens copied as-is — re-formatting full-precision periods
creates phantom records), header lines are cloned from the source's first
section (exactly 2 `#` + 6 `>` lines — the exe's fixed structure).

`-modem` (used internally by `-modem-run` too) parses ModEM data/response
files with multiple `>` sections (section
names matched with spaces/underscores normalized — `Rho_Phase` and
`Off_Diagonal_Rho_Phase` both parse): `Full_Impedance` (ZXX/ZXY/ZYX/ZYY
complex), rho/phase sections (`RHOXY/PHSXY/RHOYX/PHSYX` single-value
rows, ModEM's default inversion output; Z rebuilt as
`sqrt(ωμ0·ρa)·e^{iφ}` in SI, missing diagonal filled with 0),
`Full_Vertical_Components` (TX/TY tipper), and the modified-ModEM
`EM_Fields` section (EPolEx..BPolHz — unit X/Y-polarization
incident-field responses, zeros in the input file filled in by forward).
When non-zero `EM_Fields` values are present they become the site
fingerprint directly ("fields" path); otherwise sites are built as
H1=(1,0)/H2=(0,1), E=Z·H, Hz=T·H ("impedance" path). Stations are
keyed by (Code, X, Y) — grid surveys reusing one code get `_N` suffixes.
Each station needs ≥2 complete periods. Summary JSON gains
`"action":"modem2xmlsite"`. Defaults: SI units (impedance [mV/km]/[nT]
×1e-6/(1e-9/μ0); EM_Fields is already SI — E [V/m] passthrough, H is
magnetic flux density B [T] ×1/μ0 → A/m, so E/H equals the site impedance;
the section's `[mV/km]/[nT]` header line is boilerplate copied from the
impedance section); `-modem-units raw` keeps file units. Phases pass
through unchanged (measured ModEM output is the same +45°-style convention
as XMLSITE/EDI; `-modem-conjugate on` flips all phases if a file ever says
`exp(-iωt)`). Note: the fields and impedance sections are two evaluation
paths of the modified exe — they agree to ~1e-6 at long periods but can
diverge ~20% at the shortest periods a coarse grid cannot resolve
(COMMEMI at T=0.001 s); the fields path trusts the fields section.

Use case: turn any ModEM 3-D forward run into calibrated Phoenix
time series at the model's station sites, or feed the XMLSITE fingerprints
to multi-station separation research (snrinv).

### 8. QC 报告：逐站 ρa/φ 理论 vs 合成

```bash
synthts -qc out_mta            # 或 synthts -qc -out out_mta
```

For every site found in the synthesis output dir (`<site>_dense.XMLSITE`,
merged with plain `<site>.XMLSITE`), estimates Z over up to 40 log-spaced
anchor freqs (`mtprocess.estimate_impedance_at`, `bypass_screen=True` for
the noise engine), compares ρa/φ against the model theory
(`E·H⁻¹`), and writes `qc_<site>.png` (XY/YX theory lines vs synthetic
markers; matplotlib optional — degrades to text/JSON without it) plus
`qc_summary.json` with per-site median/max errors. Median ρa deviations
>10% print a reminder that the noise engine has a physical floor at low
freqs (CTS nearest-neighbour × continuous spectra) — lengthen `-dur`.
Report tool: always exits 0 unless no models/TS are found.

### 9. 速查与自检

```bash
synthts -info        # 频段表（TS2-5 采样率/覆盖/连续性）、体积预估、环境
synthts -selftest    # 双引擎微型合成 + f=1Hz 锚点阻抗回验；PASS 才 exit 0
```

### 10. 实测 EDI → 合成测点（⚠ 非标准流程）

```bash
synthts -edi JHMT03SB.EDI -out sites2/ [-dense 4]
```

Parses SEG 1.0 EDI (Z + tipper; EMPTY→NaN dropped; unit-polarization
fingerprint, SI conversion) into `<name>.XMLSITE`. **Mandatory warning**:
the synthesized "theory" replicates that site's *measured* response —
never use it as an unbiased ground truth for processing tests on the same
site (circular); legitimate uses are reproducible trial data with measured
statistics / controlled-noise controls. The banner prints on every run and
the summary JSON carries a `warning` field.

## Behaviour notes

- `-mode MT` → TS3/TS4/TS5; `-mode AMT` → TS2/TS3/TS4. AMT TS2 runs at 24 kHz — a 1 h
  AMT file is ~430 MB; keep `-dur` small for smoke runs.
- `-dense 0` disables densification; with dense grids use `-periods >= 128`
  (lines engine only — inert for noise with a WARNING).
- Same `-seed` → byte-identical reruns (safe to assert file hashes in tests).
- Calibration is built in (`synthts/assets/`); never pass TBL/CLB/CLC files.
- `mtprocess.process` on the emitted `<site>_dense.XMLSITE` validates the full dense grid.
- Output statistics differ by engine: `noise` data has continuous spectra,
  log-normal band envelopes (bursts) and fresh excitation per window; `lines`
  data is a segmented-modulated comb (energy only at grid frequencies).
  Choose `noise` for realistic excitation, `lines` for deterministic,
  grid-matched references (e.g. research-topic-1 fitting basis).
- Long noise runs: 72 h MT natural ≈ 2 min (TS5 whole-axis band-synthesis FFTs
  dominate); 1 h two-site ≈ 0.5 s.

## Failure modes

| Symptom | Cause / fix |
|---|---|
| exit 2 `模型无频点` / `全部 isNoise` | model has no (signal) frequencies — check the XMLSITE |
| exit 2 `测点名重复` | two models resolve to the same site name (outputs would collide) |
| exit 2 `须指定一种工作模式` / `互斥` | none or several of -model/-modem-run/-modem/-modem-add-fields/-edi/-qc given |
| exit 2 `非有限值(NaN/Inf)的频点` | model carries NaN fields (common from coarse-grid forwards at the shortest periods) — clean listed freqs |
| exit 2 `无有效测点` (modem) | no station with ≥2 complete periods — the error lists the file's components |
| exit 2 `-modem-exe` / `ModEM exe 不存在` | `-modem-run` needs the modified exe path |
| exit 2 `ModEM exe 失败` / `无回填的 EM_Fields` | see `fwd/modem_fwd.log` (tail is in the message); long Chinese paths must be pre-copied to short ASCII (auto-handled by -modem-run) |
| exit 2 other | error text on stdout with 提示 lines; rerun with `SYNTHTS_DEBUG=1` for the traceback |

## Output files per site

`<site>.TBL`, `<site>.TS2`…`<site>.TS5` (32-byte record headers, 24-bit little-endian
channel-interleaved samples), plus `<site>_dense.XMLSITE` when `-dense > 0`.
