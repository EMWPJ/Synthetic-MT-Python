# -*- coding: utf-8 -*-
"""Smoke tests: MT/AMT CLI runs via the installed package, JSON summary,
same-seed byte-level determinism."""
import hashlib
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(REPO, 'examples', 'mta.XMLSITE')


def run_cli(args, cwd):
    return subprocess.run([sys.executable, '-m', 'synthts'] + args,
                          cwd=cwd, capture_output=True, text=True,
                          encoding='utf-8', errors='replace')


def summary_of(stdout):
    lines = [l for l in stdout.splitlines() if l.startswith('SYNTHTS_SUMMARY ')]
    assert lines, 'SYNTHTS_SUMMARY line missing from stdout'
    return json.loads(lines[-1][len('SYNTHTS_SUMMARY '):])


def digest_dir(path):
    out = {}
    for name in sorted(os.listdir(path)):
        fp = os.path.join(path, name)
        if os.path.isfile(fp):
            with open(fp, 'rb') as f:
                out[name] = hashlib.sha256(f.read()).hexdigest()
    return out


def test_mt_smoke(tmp_path):
    r = run_cli(['-model', MODEL, '-out', str(tmp_path / 'mt'), '-dur', '300'],
                str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    s = summary_of(r.stdout)
    assert s['mode'] == 'MT'
    assert s['sites'] == ['MTA']
    assert s['dur_s'] == 300.0
    assert s['seed'] == 42 and s['dense'] == 4.0
    assert sorted(s['schemas']) == ['TS3', 'TS4', 'TS5']
    # TBL + three TS bands + the densified model
    assert s['n_files'] == len(s['files']) == 5
    for f in s['files']:
        assert os.path.isfile(f), f
    assert os.path.isfile(str(tmp_path / 'mt' / 'MTA.TBL'))


def test_amt_smoke(tmp_path):
    r = run_cli(['-model', MODEL, '-out', str(tmp_path / 'amt'),
                 '-mode', 'AMT', '-dur', '120'], str(tmp_path))
    assert r.returncode == 0, r.stdout + r.stderr
    s = summary_of(r.stdout)
    assert s['mode'] == 'AMT'
    assert s['dur_s'] == 120.0
    assert sorted(s['schemas']) == ['TS2', 'TS3', 'TS4']
    assert s['n_files'] == len(s['files']) == 5
    for f in s['files']:
        assert os.path.isfile(f), f


def test_same_seed_byte_reproducible(tmp_path):
    runs = []
    for k in (1, 2):
        d = tmp_path / ('det%d' % k)
        r = run_cli(['-model', MODEL, '-out', str(d), '-dur', '120',
                     '-seed', '7'], str(tmp_path))
        assert r.returncode == 0, r.stdout + r.stderr
        runs.append(digest_dir(str(d)))
    assert runs[0] == runs[1]
