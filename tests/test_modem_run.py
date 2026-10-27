# -*- coding: utf-8 -*-
"""modem_run 单测：子集（段结构/counts 重写/token 保真）+ 编排错误路径。"""
import os
import re

import pytest

from synthts.cli import main
from synthts.modem import add_em_fields, parse_modem_response
from synthts.modem_run import logspaced_idx, orchestrate, subset_dat

from tests.test_modem import _dat, _imp_rows, _tip_rows


def _write(tmp_path, text, name):
    p = tmp_path / name
    p.write_text(text, encoding='ascii', newline='')
    return str(p)


def _counts_lines(path):
    out = []
    for ln in open(path, encoding='utf-8', errors='replace'):
        if re.match(r'^\s*>\s*\d+\s+\d+\s*$', ln):
            out.append(ln.strip())
    return out


def test_logspaced_idx():
    assert logspaced_idx(10, 10) == list(range(10))
    assert logspaced_idx(10, 1) == [0]
    idx = logspaced_idx(9, 3)
    assert idx[0] == 0 and idx[-1] == 8 and len(set(idx)) == 3


def test_subset_multisection_counts(tmp_path):
    """多段文件取子集：各段 counts 行重写、行序保持、非数据行保留。"""
    rows = '\n'.join([_imp_rows(1.0E-3, 'S1'), _imp_rows(1.0E-2, 'S1'),
                      _imp_rows(1.0E-3, 'S2'), _imp_rows(1.0E-2, 'S2')])
    tip = '\n'.join([_tip_rows(1.0E-3, 'S1'), _tip_rows(1.0E-2, 'S1'),
                     _tip_rows(1.0E-3, 'S2'), _tip_rows(1.0E-2, 'S2')])
    src = _write(tmp_path, _dat(imp_row=rows, tip_row=tip), 'in.dat')
    dst = str(tmp_path / 'sub.dat')
    info = subset_dat(src, dst, stations=['S1'])
    assert info == {'n_periods': 2, 'n_stations': 1, 'rows': 12}
    assert _counts_lines(dst) == ['>       2     1'] * 2   # 每段各重写
    txt = open(dst, encoding='ascii').read()
    assert 'S2' not in txt
    assert txt.count('ZXY') == 2 and txt.count('TX') == 2
    assert '>  Full_Impedance' in txt and \
        '>  Full_Vertical_Components' in txt


def test_subset_max_periods(tmp_path):
    rows = '\n'.join([_imp_rows(p, 'S1') for p in
                      (1.0E-3, 1.0E-2, 1.0E-1, 1.0, 10.0)])
    src = _write(tmp_path, _dat(imp_row=rows), 'in.dat')
    dst = str(tmp_path / 'sub.dat')
    info = subset_dat(src, dst, max_periods=2)
    assert info['n_periods'] == 2 and info['rows'] == 8
    r = parse_modem_response(dst)
    assert r.periods == [1.0e-3, 10.0]               # 等距取含首尾


def test_full_precision_period_no_phantom(tmp_path):
    """全精度周期 token 保真：补段后不产生幻影 (周期,站) 记录。"""
    rows = '\n'.join([
        '0.888888888  S1  30.0 113.0  -1000.0 2000.0 0.000  %s  '
        '1.0 1.0  2.0E+15' % c for c in ('ZXX', 'ZXY', 'ZYX', 'ZYY')] +
        [_imp_rows(1.0E-2, 'S1')])
    src = _write(tmp_path, _dat(imp_row=rows), 'in.dat')
    dst = str(tmp_path / 'p.dat')
    add_em_fields(src, dst)
    r = parse_modem_response(dst)
    assert len(r.record_order) == 2                   # 2 周期 × 1 站，无幻影
    assert r.periods == pytest.approx([0.01, 0.888888888])
    em = [ln for ln in open(dst, encoding='ascii')
          if 'EPolEx' in ln and '0.888888888' in ln]
    assert em and em[0].split()[0] == '0.888888888'   # 周期原文照抄


def test_orchestrate_refusals(tmp_path):
    rows = '\n'.join([_imp_rows(1.0E-3, 'S1'), _imp_rows(1.0E-2, 'S1')])
    dat = _write(tmp_path, _dat(imp_row=rows), 'in.dat')
    filled = str(tmp_path / 'filled.dat')
    add_em_fields(dat, filled)
    out = str(tmp_path / 'o')
    with pytest.raises(ValueError, match='-modem-exe'):
        orchestrate('', 'm.modem', dat, out)           # exe 缺失
    with pytest.raises(ValueError, match='已含 EM_Fields'):
        orchestrate('whatever', 'm.modem', filled, out)  # 非常规 dat
    up_down = os.path.join(os.pardir, 'up.dat')
    with pytest.raises(ValueError, match='路径不得包含'):
        subset_dat(up_down, str(tmp_path / 'x.dat'))


def test_cli_modem_run_requires_exe(tmp_path, capsys):
    rc = main(['-modem-run', 'm.modem', 'd.dat',
               '-out', str(tmp_path / 'o')])
    assert rc == 2
    assert '-modem-exe' in capsys.readouterr().out
