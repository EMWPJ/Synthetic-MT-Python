# -*- coding: utf-8 -*-
"""ModEM 格式兼容单测：多段解析 / RhoPhase 重建 / 场值路径 / 相位与单位 / 路径守卫。"""
import math
import os

import pytest

from synthts.modem import (MU0, ModemResponse, _EM_COMPS,
                           add_em_fields, parse_modem_response,
                           sites_from_response, write_modem_sites)
from synthts.model import load_site


def _dat(imp_row=None, tip_row=None, fld_row=None, counts='2     2',
         exp='>  exp(i\\omega t)'):
    """拼一个多段 ModEM 文本（段头行序与真实文件一致：段名→exp→单位）。"""
    head = ['# test', '# Period(s) Code GG_Lat GG_Lon X(m) Y(m) Z(m) '
            'Component Real Imag Error']
    secs = []
    if imp_row:
        secs += head + ['>  Full_Impedance', exp, '>  [mV/km]/[nT]',
                        '>     0.00', '>    0.000   0.000', '>   ' + counts,
                        imp_row]
    if tip_row:
        secs += head + ['>  Full_Vertical_Components', exp, '>  []',
                        '>     0.00', '>    0.000   0.000', '>   ' + counts,
                        tip_row]
    if fld_row:
        secs += head + ['>  EM_Fields', exp, '>  [mV/km]/[nT]', '>     0.00',
                        '>    0.000   0.000', '>   ' + counts, fld_row]
    return '\n'.join(secs) + '\n'


def _imp_rows(per, code, zxy=(166.4968, 192.9013), zyx=(-130.0816, -147.4482),
              elev=0.0):
    lat, lon, x, y = 30.0, 113.0, -1000.0, 2000.0
    return '\n'.join(
        '%.6E  %s  %.7f %.7f  %.1f %.1f %.3f  %s  %.6E  %.6E  2.0E+15'
        % (per, code, lat, lon, x, y, elev, comp, re_, im_)
        for comp, (re_, im_) in (('ZXX', (1.0, -1.0)),
                                 ('ZXY', zxy),
                                 ('ZYX', zyx),
                                 ('ZYY', (2.0, 0.5))))


def _tip_rows(per, code):
    return '\n'.join(
        '%.6E  %s  30.0 113.0  -1000.0 2000.0 0.000  %s  %.6E  %.6E  2.0E+15'
        % (per, code, comp, re_, im_)
        for comp, (re_, im_) in (('TX', (0.1, 0.2)), ('TY', (0.3, 0.4))))


def _write(tmp_path, text, name='m.dat'):
    p = tmp_path / name
    p.write_text(text, encoding='ascii')
    return str(p)


def test_parse_multisection_impedance(tmp_path):
    """多段文件：阻抗+倾子段 → impedance 路径；不完整站位被剔除。"""
    rows = '\n'.join([_imp_rows(1.0E-3, 'S1'), _imp_rows(1.0E-2, 'S1'),
                      _imp_rows(1.0E-3, 'S2')])
    tip = '\n'.join([_tip_rows(1.0E-3, 'S1'), _tip_rows(1.0E-2, 'S1')])
    path = _write(tmp_path, _dat(imp_row=rows, tip_row=tip))
    r = parse_modem_response(path)
    assert r.fmt_z == 'full_impedance' and not r.rhophase
    assert r.periods == [1.0e-3, 1.0e-2]
    assert r.conversion_path == 'impedance'
    keys = r.station_keys()
    assert [k[0] for k in keys] == ['S1', 'S2']
    assert r.coords[keys[0]] == (30.0, 113.0, 0.0)
    sites = sites_from_response(r)
    assert [s.name for s in sites] == ['S1']            # S2 仅 1 频点 → 剔除
    s1 = sites[0]
    assert len(s1.fields) == 2
    f0 = s1.fields[0]
    assert f0.frequency == pytest.approx(1000.0)
    # 单位换算：SI = 原值 × 1e-6/(1e-9/μ0)；H1=(1,0) → ex1=ZXX·fac
    fac = 1e-6 / (1e-9 / MU0)
    assert f0.ex1 == pytest.approx(complex(1.0, -1.0) * fac)
    assert f0.ey1 == pytest.approx(complex(-130.0816, -147.4482) * fac)
    assert f0.hx1 == 1.0 + 0j and f0.hy2 == 1.0 + 0j
    assert f0.hz1 == pytest.approx(complex(0.1, 0.2))   # 倾子无量纲，不换算
    assert f0.hz2 == pytest.approx(complex(0.3, 0.4))


def test_rhophase_reconstruction(tmp_path):
    """RhoPhase 段：Z = sqrt(ωμ0·ρa)·e^{iφ}（重建即 SI，不再换算）。"""
    rho, ph = 100.0, 45.0
    rows = '\n'.join(
        '%.6E  S1  30.0 113.0  -1000.0 2000.0 0.000  %s  %.6E  %.6E  2.0E+15'
        % (per, comp, rho, ph)
        for per in (0.01, 0.02)
        for comp in ('ZXX', 'ZXY', 'ZYX', 'ZYY'))
    path = _write(tmp_path,
                  _dat(imp_row=rows).replace('>  Full_Impedance',
                                             '>  RhoPhase'))
    r = parse_modem_response(path)
    assert r.rhophase
    site = sites_from_response(r)[0]
    assert len(site.fields) == 2
    z_expect = math.sqrt(2 * math.pi * 100.0 * MU0 * rho) * complex(
        math.cos(math.radians(ph)), math.sin(math.radians(ph)))
    assert site.fields[0].ex1 == pytest.approx(z_expect, rel=1e-12)
    assert site.fields[1].ex1 == pytest.approx(
        z_expect / math.sqrt(2.0), rel=1e-12)        # 周期翻倍 → ω 减半 → |Z|/√2


def test_fields_path_and_units(tmp_path):
    """EM_Fields 段非零 → fields 路径，10 复数直取（SI 换算）。"""
    vals = dict(EPOLEX=(1.0, 2.0), EPOLEY=(3.0, -4.0), EPOLHX=(0.5, 0.25),
                EPOLHY=(0.0, 0.125), EPOLHZ=(0.0, 0.0),
                BPOLEX=(-1.0, 0.5), BPOLEY=(2.0, 2.0), BPOLHX=(1.0, 1.0),
                BPOLHY=(0.25, 0.0), BPOLHZ=(0.0, -1.0))
    rows = '\n'.join(
        '%.6E  S1  30.0 113.0  -1000.0 2000.0 0.000  %s  %.6E  %.6E  2.0E+15'
        % (per, comp, re_, im_)
        for per in (1.0E-3, 1.0E-2) for comp, (re_, im_) in vals.items())
    path = _write(tmp_path, _dat(fld_row=rows))
    r = parse_modem_response(path)
    assert r.fields_present and r.fields_nonzero
    assert r.conversion_path == 'fields'
    site = sites_from_response(r)[0]
    assert site.name == 'S1' and len(site.fields) == 2
    f0 = site.fields[0]
    assert f0.frequency == pytest.approx(1000.0)
    e_fac, h_fac = 1e-6, 1e-9 / MU0
    assert f0.ex1 == pytest.approx(complex(1.0, 2.0) * e_fac)
    assert f0.hz2 == pytest.approx(complex(0.0, -1.0) * h_fac)


def test_conjugate_auto_passthrough_and_forced(tmp_path):
    """auto：exp(iωt)/exp(+iωt) 直通（实测 +45° 口径）；exp(-i 共轭；on 强制。"""
    rows = _imp_rows(1.0E-3, 'S1') + '\n' + _imp_rows(1.0E-2, 'S1')
    r = parse_modem_response(_write(
        tmp_path, _dat(imp_row=rows, exp='>  exp(+i\\omega t)'),
        name='a.dat'))
    z0 = sites_from_response(r)[0].fields[0].ey1
    assert z0.real == pytest.approx(-130.0816 * 1e-6 / (1e-9 / MU0))
    assert z0.imag < 0                                # 直通：虚部保持文件原值
    r2 = parse_modem_response(_write(
        tmp_path, _dat(imp_row=rows, exp='>  exp(-i\\omega t)'),
        name='b.dat'))
    z1 = sites_from_response(r2)[0].fields[0].ey1
    assert z1.imag > 0                                # 段头明写 -i → 共轭
    z2 = sites_from_response(r, conjugate=True)[0].fields[0].ey1
    assert z2 == pytest.approx(z0.conjugate())
    assert z2 == pytest.approx(z1)


def test_same_code_multi_coord_naming(tmp_path):
    """同码多坐标（COMMEMI 网格式）→ 站名加序号后缀。"""
    a = _imp_rows(1.0E-3, '001') + '\n' + _imp_rows(1.0E-2, '001')
    b = _imp_rows(1.0E-3, '001').replace('-1000.0', '-3000.0') + '\n' + \
        _imp_rows(1.0E-2, '001').replace('-1000.0', '-3000.0')
    r = parse_modem_response(_write(tmp_path, _dat(imp_row=a + '\n' + b)))
    assert [s.name for s in sites_from_response(r)] == ['001_1', '001_2']


def test_write_and_roundtrip(tmp_path):
    """落盘 XMLSITE 可被 load_site 读回，场值一致；同名站拒绝。"""
    rows = _imp_rows(1.0E-3, 'S1') + '\n' + _imp_rows(1.0E-2, 'S1')
    r = parse_modem_response(_write(tmp_path, _dat(imp_row=rows),
                                    name='rt.dat'))
    sites = sites_from_response(r)
    out = str(tmp_path / 'sites')
    paths = write_modem_sites(sites, out)
    assert len(paths) == 1 and os.path.isfile(paths[0])
    back = load_site(paths[0])
    assert back.name == 'S1'
    assert back.fields[0].ex1 == pytest.approx(sites[0].fields[0].ex1)
    with pytest.raises(ValueError, match='重复'):
        write_modem_sites(sites + [sites[0]], str(tmp_path / 'dup'))


def test_path_and_input_guards(tmp_path):
    """读写路径拒 '..'；坏行静默跳过；缺分量的频点不成锚。"""
    with pytest.raises(ValueError, match='\\.\\.'):
        parse_modem_response('../evil.dat')
    with pytest.raises(ValueError, match='\\.\\.'):
        write_modem_sites([], str(tmp_path / 'a' / '..' / 'b'))
    bad = '\n'.join([_imp_rows(1.0E-3, 'S1'),
                     'garbage line 1 2 3',
                     _imp_rows(1.0E-2, 'S1').replace('ZXY', 'ZQQ')
                     .replace('ZYX', 'ZQQ'),
                     _imp_rows(1.0E-1, 'S1')])
    r = parse_modem_response(_write(tmp_path, _dat(imp_row=bad),
                                    name='g.dat'))
    assert r.periods == [1.0e-3, 1.0e-2, 1.0e-1]      # 坏行跳过，周期仍收集
    sites = sites_from_response(r)
    assert [s.name for s in sites] == ['S1']
    assert [f.frequency for f in sites[0].fields] == \
        pytest.approx([1000.0, 10.0])                 # 1e-2 缺 ZXY/ZYX 被跳过


def test_empty_response_rejected(tmp_path):
    p = _write(tmp_path, '# nothing here\n', name='empty.dat')
    r = parse_modem_response(p)
    assert r.data == {} and r.periods == []


def test_add_em_fields_section(tmp_path):
    """补 EM_Fields 段：源内容逐字节保留 + 追加规范占位段（站位主序）。"""
    rows = '\n'.join([_imp_rows(1.0E-3, 'S1'), _imp_rows(1.0E-2, 'S1'),
                      _imp_rows(1.0E-3, 'S2'), _imp_rows(1.0E-2, 'S2')])
    tip = '\n'.join([_tip_rows(1.0E-3, 'S1'), _tip_rows(1.0E-2, 'S1')])
    src = _write(tmp_path, _dat(imp_row=rows, tip_row=tip), name='in.dat')
    src_bytes = open(src, 'rb').read()
    dst = str(tmp_path / 'out.dat')
    info = add_em_fields(src, dst)
    assert info['n_periods'] == 2 and info['n_stations'] == 2
    assert info['n_records'] == 4 and info['n_rows'] == 40
    out = open(dst, 'rb').read()
    assert out[:len(src_bytes)] == src_bytes            # 源内容逐字节保留
    lines = out[len(src_bytes):].decode('ascii').splitlines()
    assert lines[2] == '>  EM_Fields'
    assert lines[7].lstrip('>').split() == ['2', '2']               # >%8d%6d 计数行
    data = lines[8:]
    assert len(data) == 40
    rec0 = data[0].split()
    assert rec0[7] == 'EPolEx' and rec0[8] == '0.000000E+00'
    assert [ln.split()[7] for ln in data[:10]] == list(_EM_COMPS)
    # 枚举序跟随源文件：S1(1e-3, 1e-2) → S2(1e-3, 1e-2)
    order = [(ln.split()[0], ln.split()[1]) for ln in data[::10]]
    assert order == [('1.000000E-03', 'S1'), ('1.000000E-02', 'S1'),
                     ('1.000000E-03', 'S2'), ('1.000000E-02', 'S2')]


def test_add_em_fields_elev_and_periodmajor(tmp_path):
    """高程列保真；周期主序源文件 → 生成段同为周期主序。"""
    rows = '\n'.join([_imp_rows(1.0E-3, 'S1', elev=12.5),
                      _imp_rows(1.0E-3, 'S2', elev=12.5),
                      _imp_rows(1.0E-2, 'S1', elev=12.5),
                      _imp_rows(1.0E-2, 'S2', elev=12.5)])
    src = _write(tmp_path, _dat(imp_row=rows), name='pm.dat')
    dst = str(tmp_path / 'pm_out.dat')
    add_em_fields(src, dst)
    lines = open(dst, 'rb').read().decode('ascii').splitlines()
    tail = lines[lines.index('>  EM_Fields') + 5:]
    assert tail[0].lstrip('>').split() == ['2', '2']                # 计数行
    data = tail[1:]
    assert data[0].split()[6] == '12.500'               # 高程列保真
    order = [(ln.split()[0], ln.split()[1]) for ln in data[::10]]
    assert order == [('1.000000E-03', 'S1'), ('1.000000E-03', 'S2'),
                     ('1.000000E-02', 'S1'), ('1.000000E-02', 'S2')]


def test_add_em_fields_refusals(tmp_path):
    """已含 EM_Fields 段 / src==dst / '..' 路径 一律拒绝。"""
    rows = '\n'.join([_imp_rows(1.0E-3, 'S1'), _imp_rows(1.0E-2, 'S1')])
    src = _write(tmp_path, _dat(imp_row=rows), name='r.dat')
    dst = str(tmp_path / 'r2.dat')
    add_em_fields(src, dst)
    with pytest.raises(ValueError, match='已含'):
        add_em_fields(dst, str(tmp_path / 'r3.dat'))    # 产物再补 → 拒绝
    with pytest.raises(ValueError, match='不同'):
        add_em_fields(src, src)
    with pytest.raises(ValueError, match='\\.\\.'):
        add_em_fields(src, str(tmp_path / 'a' / '..' / 'b.dat'))
    with pytest.raises(ValueError, match='\\.\\.'):
        add_em_fields('../x.dat', str(tmp_path / 'y.dat'))
