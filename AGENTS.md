# AGENTS.md

Guidance for AI coding agents working in this repository (Synthetic-MT-Python /
SMTTS). Read this before changing anything under `synthts/`.

## What this is

Python re-implementation of the MTDataPro `SynthTS` synthesizer: it generates
Phoenix-format magnetotelluric synthetic time series (`.TBL` + `.TS2…TS5`) from
`.XMLSITE` forward-model sites, with the instrument calibration embedded.
Two synthesis engines share one writer:

- **lines** (v1) — sparse comb matched to the processing window; byte-parity
  reference engine, pinned by the golden bucket. Change nothing here casually.
- **noise** (v2, default) — three-layer natural-source model (white floor +
  ½-octave amplitude-complementary bands with log-normal gain-segment chains +
  Poisson Gabor events), rendered per record band (`render_fft`).

## Commands

```bash
pip install -e ".[plot]"            # miniconda python; numpy version matters (golden buckets)
python -m pytest                    # full suite (~9 min; golden + gates + smoke)
python -m pytest -q -k "not slow"   # fast subset
python tools/mt_e2e_demo.py --source both   # dual-engine gates; must print ALL PASS
python tools/plot_demo.py           # MT paper figures  -> ./plot_out/
python tools/plot_demo_amt.py       # AMT paper figures -> ./plot_out_amt/
```

An accepted change requires: pytest green **and** `mt_e2e_demo --source both`
ALL PASS. The demo output dirs (`mt_e2e_out_*`, `plot_out*`, `run48h/`) are
git-ignored scratch.

## Architecture

```
synthts/
├── cli.py            # CLI; per-mode TBL/assets; JSON summary last line of stdout
├── synth.py          # band schemas (build_schemas), segment schedules, lines engine
├── render_fft.py     # v2 record-band renderer; transfer_matrix (T = responds x fields)
├── noise_source.py   # L1/L2/L3/L4 source model; render_schema
├── bands.py          # ½-octave amplitude-complementary filter bank (public API)
├── mtprocess.py      # processing twin: window_dft, estimate_impedance_at,
│                     #   process_fft_bins (render-isomorphic theory reference)
├── calib.py          # CTS calibration (nearest / smooth), transform_cts, band_of_freq
├── cts_mt/cts_amt    # embedded sparse calibration tables (bands 3/4/5 and 2-5)
├── cts_dense.py      # dense CTS (syscal_dll-generated; tools/gen_dense_cts.py)
├── phoenix_ts.py     # TBL + TS binary writer (Phoenix byte format)
├── modem.py          # ModEM data/response → XMLSITE sites (multi-section:
│                     #   Full_Impedance / RhoPhase / tipper / EM_Fields);
│                     #   add_em_fields appends an EM_Fields placeholder
│                     #   section for the modified exe's forward runs
└── source_params.py  # noise presets + L4 schema validation
```

Processing (`mtprocess`) is the **validation twin** of synthesis: the e2e demo
feeds synthesized files back through it and compares against the render-isomorphic
theory. Any render change usually needs the matching processing-side change.

## Conventions you must not break

- **Band layout**: calibration bands come from `calib.band_of_freq`
  (>400→2, >8→3, >0.8→4, else 5). File schemas come from `synth.build_schemas`:
  MT = TS3/TS4/TS5 with TS5 continuous; AMT = TS2/TS3/TS4 with **TS4 continuous
  from 0.01 Hz** (`TS4_FRE_AMT`; MT's windowed TS4 keeps its 0.1 Hz floor).
  `mtprocess.TS_SPEC` / `TS_SPEC_AMT` mirror these for file routing.
- **Electric-channel sign**: forward fields are E = Z·H in the e^(−iωt)
  convention; impedance estimation in the processing chain follows e^(+iωt).
  The writer therefore flips Ex/Ey on disk (`phoenix_ts._pack_body`) and
  `mtprocess` negates them again after calibration (both `estimate_impedance_at`
  and `process_fft_bins`). The double negation keeps the round trip sign-neutral
  while files carry the processing-side convention. Do not "fix" either side alone.
- **Golden bucket**: `tests/golden/<numpy-version>/hashes.json` pins the lines
  engine byte-for-byte. Intentional convention changes require regenerating the
  bucket and saying so in the commit message; nothing else may move it.
- **JSON summary** is the last stdout line only, never in `-log`; file paths use
  forward slashes.
- **Determinism**: same seed → byte-identical output. Never introduce per-run
  normalization or wall-clock-dependent values into the render path.
- **Writers must reject `..` path traversal** (see `write_time_series`); keep it
  that way in any new output path.

## Known physics/engineering traps (all were real bugs once)

- Gapped bands (TS2/TS3 in AMT, TS4 in MT): processing windows must not cross
  record boundaries — use `hop=nfft` aligned to record length; continuous bands
  may use 50 % overlap.
- `MIN_WIN_PERIODS = 4`: bins whose period exceeds a quarter of the window are
  skipped; when picking nfft, make key anchors land on integer bins
  (`f·nfft/fs ∈ ℤ`).
- Quantization: rendered counts use a fixed deterministic gain
  (`COUNTS_SCALE = 256`) with a clip guard. Per-file peak normalization is
  forbidden (breaks cross-band/cross-seed consistency).
- Dense CTS rows with an exactly-zero channel are syscal_dll out-of-range
  evaluations (e.g. Hx ≈ 12 kHz) — `calib._dense_group` drops them;
  `tools/gen_dense_cts.py` filters them at generation time.
- `SMOOTH_RESPONDS` (calib module flag, default False) switches responds from
  nearest-neighbour to dense-table interpolation. Render and process must run
  in the same mode; demo scripts set it once at import, the shipped default
  path keeps both sides False.
- MT phases are folded to (−90°, 90°] for display; a pure sign flip of Z is
  invisible after folding, a convention conjugation is not.
- ModEM output phases are empirically the same +45°-style convention as
  XMLSITE/EDI (measured on the EMWPJ COMMEMI3D-2A example: folded ZYX phase
  +48.6°, fields section consistent with the impedance section) — even though
  the file header loosely says `exp(iωt)`. `synthts.modem` passes phases
  through by default; do not conjugate on the strength of that header line.
- The modified-ModEM exe parses dat sections with a FIXED structure: each
  section is exactly 2 `#` comment lines + 6 `>` header lines + data rows.
  A third comment line is consumed as the data-type declaration (fails with
  `Unknown data type: Period(s) …`, truncated at a fixed width) — this is
  what broke the Jianghan prep before; `add_em_fields` emits exactly 2.
- Odd/even record lengths: never infer n from the rfft length (`n ≡ 0 mod 4`
  is ambiguous) — pass n explicitly.

## Output files (per site, in `-out`)

`<site>.TBL`, `<site>.TS2…TS5` (24-bit LE, channel-interleaved, 32-byte record
headers), `<site>_dense.XMLSITE` when `-dense > 0`. The time span is pinned to
2050-01-01 00:00:00 + `-dur`.
