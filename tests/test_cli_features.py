# -*- coding: utf-8 -*-
"""CLI 新功能单测：-info / -selftest / 模式互斥 / 详细错误 / NaN 预检。"""
import json

import numpy as np

from synthts.cli import main
from synthts.model import Field, ForwardSite, save_site


def test_info(capsys):
    assert main(['-info']) == 0
    out = capsys.readouterr().out
    for token in ('TS5', 'TS2', '24000', '连续', '间歇', 'MT', 'AMT',
                  'selftest', '250 MB'):
        assert token in out


def test_selftest(capsys):
    assert main(['-selftest']) == 0
    out = capsys.readouterr().out
    assert '[lines] PASS' in out and '[noise] PASS' in out
    assert 'SELFTEST PASS' in out
    line = [ln for ln in out.splitlines()
            if ln.startswith('SYNTHTS_SUMMARY ')][-1]
    j = json.loads(line[len('SYNTHTS_SUMMARY '):])
    assert j['action'] == 'selftest' and j['passed'] is True


def test_mode_exclusive_error(capsys):
    rc = main(['-model', 'a.XMLSITE', '-modem', 'b.dat', '-out', 'o'])
    assert rc == 2
    assert '互斥' in capsys.readouterr().out


def test_no_mode_error(capsys):
    rc = main(['-out', 'o'])
    assert rc == 2
    out = capsys.readouterr().out
    assert '工作模式' in out and '-modem-run' in out


def test_missing_out_error(capsys, tmp_path):
    rc = main(['-model', 'x.XMLSITE'])
    assert rc == 2
    assert '-out' in capsys.readouterr().out


def test_detailed_error_missing_file(capsys, tmp_path):
    rc = main(['-model', str(tmp_path / 'nope.XMLSITE'),
               '-out', str(tmp_path / 'o')])
    assert rc == 2
    out = capsys.readouterr().out
    assert 'FileNotFoundError' in out
    assert '提示' in out and '文件不存在' in out


def test_nan_model_rejected(capsys, tmp_path):
    site = ForwardSite()
    site.name = 'NAN1'
    for i, f in enumerate((1.0, 10.0, 100.0)):
        fld = Field(frequency=f, is_noise=False)
        fld.hx1, fld.hy1 = 1.0 + 0j, 0.0j
        fld.hx2, fld.hy2 = 0.0j, 1.0j
        z = 1.0 + 1.0j if i != 1 else complex(float('nan'), 0.0)
        fld.ex1, fld.ey1 = 0.0j, -z
        fld.ex2, fld.ey2 = z, 0.0j
        site.fields.append(fld)
    m = tmp_path / 'nan.XMLSITE'
    save_site(site, str(m))
    rc = main(['-model', str(m), '-out', str(tmp_path / 'o')])
    assert rc == 2
    out = capsys.readouterr().out
    assert '非有限值' in out and '10' in out   # 指出坏频点


def test_qc_flag_without_dir(capsys):
    rc = main(['-qc'])
    assert rc == 2
    assert '目录' in capsys.readouterr().out
