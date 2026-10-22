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
| exit 2 other | error text on stdout; rerun with `SYNTHTS_DEBUG=1` for the traceback |

## Output files per site

`<site>.TBL`, `<site>.TS2`…`<site>.TS5` (32-byte record headers, 24-bit little-endian
channel-interleaved samples), plus `<site>_dense.XMLSITE` when `-dense > 0`.
