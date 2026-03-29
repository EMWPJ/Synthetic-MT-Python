# SyntheticMT Project Agents

This document describes the agent-based workflow for the SyntheticMT project.

## Project Overview

SyntheticMT is a Python library for synthesizing magnetotelluric (MT) time series based on forward modeling.

## Repository

- **GitHub**: https://github.com/EMWPJ/Synthetic-MT-Python
- **Original Delphi**: https://github.com/EMWPJ/SyntheticMTTimeSeries

## Project Structure

```
Synthetic-MT-Python/
├── src/                          # Core library
│   └── synthetic_mt/             # Modular core
│       ├── domain/               # Domain entities & services
│       ├── infrastructure/       # I/O handlers
│       ├── application/          # Use cases
│       └── presentation/         # GUI
├── examples/
│   ├── mt_workflow/              # 1D MT workflow (main focus)
│   │   ├── backend/              # Backend (algorithm + API)
│   │   │   ├── core.py          # Algorithm core
│   │   │   ├── api.py           # API interface
│   │   │   └── verify_all.py    # Verification script
│   │   ├── gui_simple.py        # Simplified GUI (frontend)
│   │   └── ...
│   └── test_synthetic.py
└── docs/
    └── plans/                    # Project plans
```

## Architecture: Backend-Frontend Separation

### Backend (`examples/mt_workflow/backend/`)

**Purpose**: Independent algorithm modules, verifiable without GUI.

**Modules**:
- `core.py` - Algorithm implementations
  - `MT1DModel` - 1D layered earth model
  - `MT1DForward` - 1D forward calculation (recursive algorithm)
  - `TimeSeriesSynthesizer` - Time series synthesis
  - `TimeSeriesProcessor` - FFT processing, impedance estimation
  - `Model1DValidator` - 1D model feature validation
- `api.py` - Unified API
  - `MTWorkflowAPI` - Singleton API class
  - `get_api()` - Access the singleton
- `verify_all.py` - Verification script

**Running Verification**:
```bash
cd examples/mt_workflow
python backend/verify_all.py
```

### Frontend (`examples/mt_workflow/gui_simple.py`)

**Purpose**: Pure UI layer, calls backend API only.

- User interaction
- Data visualization (matplotlib charts)
- Calls `backend.api.get_api()` for all computations

**Running GUI**:
```bash
cd examples/mt_workflow
python gui_simple.py
```

## 1D MT Workflow Components

### Synthesis Methods

Three synthesis methods are implemented:

#### 1. Deterministic Synthesis (`synthesize_time_series_deterministic`)
- Preserves exact E-H phase relationship: `Ex = Zxy * Hy`
- Used for algorithm verification
- Processing results match forward exactly (<2% error)

#### 2. Random Synthesis (`synthesize_time_series_random`)
- Implements paper's `RANDOM_SEG_PARTIAL` algorithm
- Per-segment random amplitude (`RandG(1,1)`) and phase (`Random * 2π`)
- Per-segment polarization angle variation: `θ ∈ [0, 2π)`
- TE/TM mode mixing: `ex = ex_TM * cos(θ) + ex_TE * sin(θ)`
- Simulates natural source polarization ellipse rotation
- Processing results have larger errors (expected behavior)

#### 3. Synthesis Classes (in `core.py`)
- `DeterministicTimeSeriesSynthesizer` - Deterministic synthesis with fixed E-H phase
- `RandomSegmentTimeSeriesSynthesizer` - Random segment synthesis with polarization variation
- `TimeSeriesSynthesizer` - Wrapper using `synthetic_mt` library

### Preset Models

| Model | Resistivity | Thickness | Description |
|-------|-------------|-----------|-------------|
| `uniform_100` | [100] Ω·m | - | Uniform halfspace |
| `uniform_1000` | [1000] Ω·m | - | Uniform halfspace |
| `two_layer_hl` | [1000, 10] Ω·m | [100] m | High-low |
| `two_layer_ll` | [10, 1000] Ω·m | [100] m | Low-high |
| `three_layer_hll` | [1000, 10, 100] Ω·m | [50, 200] m | High-low-low |

### Acquisition Systems (TS configs)

| System | Sample Rate | Frequency Range | Period Range |
|--------|-------------|-----------------|--------------|
| TS3 | 2400 Hz | 1-1000 Hz | 1ms-1s |
| TS4 | 150 Hz | 0.1-10 Hz | 0.1-10s |
| TS5 | 15 Hz | 1e-6-1 Hz | 1-1e6 s |

## Agent Workflow

When working on this project:

1. **Backend changes** (`backend/core.py`, `backend/api.py`):
   - Verify independently with `verify_all.py`
   - Do not require GUI

2. **Frontend changes** (`gui_simple.py`):
   - Only UI layer, calls existing API
   - No business logic

3. **Integration**:
   - GUI calls API which calls core
   - Any layer can be tested independently

## Key Files

| File | Purpose |
|------|---------|
| `backend/core.py` | Algorithm implementations |
| `backend/api.py` | API interface |
| `backend/verify_all.py` | Module verification |
| `gui_simple.py` | Simplified GUI |
| `src/synthetic_mt/` | Core synthesis library |

## Verification Results

### Backend Modules Verification
```
verify_constants:     PASS - MU0 correct
verify_model_creation: PASS - Halfspace & layered models
verify_ts_config:     PASS - TS3/TS4/TS5 configs
verify_halfspace_forward: PASS - rho_a=100, phase=45°
verify_layered_forward: PASS - Long period → basement resistivity
verify_1d_validation:  PASS - Zxy=-Zyx, Zxx=Zyy=0
verify_processor:      PASS - FFT frequency detection
verify_api_workflow:   PASS - Full workflow
```

### Full Pipeline Test (Forward → Synthesis → Processing → Comparison)

| Model | Max rho_a Error | Mean rho_a Error | Max Phase Error | Mean Phase Error | Status |
|-------|---------------|-----------------|-----------------|-----------------|--------|
| uniform_100 | 0.93% | 0.30% | 0.39° | 0.10° | PASS |
| uniform_1000 | 0.93% | 0.30% | 0.39° | 0.10° | PASS |
| two_layer_hl | 1.14% | 0.35% | 0.48° | 0.12° | PASS |
| two_layer_ll | 0.32% | 0.12% | 0.11° | 0.03° | PASS |
| three_layer_hll | 0.60% | 0.24% | 0.21° | 0.06° | PASS |

**Note**: Errors are due to FFT numerical precision. The deterministic synthesis preserves the E-H phase relationship, enabling accurate impedance recovery.

### Generated Test Artifacts

- `docs/plots/<model_name>_comparison.png` - Per-model forward vs processed comparison
- `docs/plots/all_models_summary.png` - All models summary plot
- `docs/plots/error_summary.png` - Error analysis plot
- `docs/plots/test_report.md` - Test report
