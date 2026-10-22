# -*- coding: utf-8 -*-
"""Minimal Python-API example: synthesize one site, process it, compare Z.

Run from the repository root after `pip install -e .`:

    python examples/example_mtprocess.py
"""
import os
import tempfile

from synthts import mtprocess
from synthts.cli import main as synth_main

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = os.path.join(HERE, 'mta.XMLSITE')


def main():
    out = tempfile.mkdtemp(prefix='synthts_example_')
    print('output dir:', out)

    # 1. Synthesize 1 h of MT data (sparse grid; parameters mirror the e2e demo
    #    so the <1e-2 anchor tolerance below is the verified one).
    rc = synth_main(['-model', MODEL, '-out', out, '-dur', '3600',
                     '-periods', '64', '-seed', '42', '-dense', '0'])
    assert rc == 0, 'synthesis failed'

    # 2. Process: per-frequency windowed DFT -> calibrate -> least-squares Z.
    rows = mtprocess.process(MODEL, out)

    # 3. Compare against the theoretical impedance of the model.
    print('  freq[Hz]     nwin   max |Z| rel.err')
    for freq, z_est, z_theory, nwin, rel_err in rows:
        print('  %-12.4g %-7d %.3e' % (freq, nwin, rel_err))
    worst = max(r[4] for r in rows)
    print('worst relative error: %.3e (sparse-grid tolerance: 1e-2)' % worst)
    return 0 if worst < 1e-2 else 1


if __name__ == '__main__':
    raise SystemExit(main())
