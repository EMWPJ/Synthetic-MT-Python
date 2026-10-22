# -*- coding: utf-8 -*-
"""Paper-style figure demo test (time series / spectrum / rho-phase)."""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.slow
def test_plot_demo(tmp_path):
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, 'tools', 'plot_demo.py')],
        cwd=str(tmp_path), capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=2400)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    for name in ('fig_time_series.png', 'fig_rho_phase.png'):
        p = tmp_path / 'plot_out' / name
        assert p.is_file() and p.stat().st_size > 10000, name
