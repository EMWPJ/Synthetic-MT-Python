# -*- coding: utf-8 -*-
"""Four-gate end-to-end test: wrapper around tools/mt_e2e_demo.py.

The demo must run with a relative output name (it rejects absolute paths),
hence cwd=tmp_path and the 'e2e_out' argument.
"""
import os
import subprocess
import sys

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.mark.slow
def test_four_gates(tmp_path):
    r = subprocess.run(
        [sys.executable, os.path.join(REPO, 'tools', 'mt_e2e_demo.py'), 'e2e_out'],
        cwd=str(tmp_path), capture_output=True, text=True,
        encoding='utf-8', errors='replace', timeout=2400)
    assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-2000:]
