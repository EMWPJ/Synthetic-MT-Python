# -*- coding: utf-8 -*-
"""EDI 入料单测：解析 / 单位 / 测点构造 / 强制警告（非标准流程）。"""
import json
import math
import os

import numpy as np
import pytest

from synthts.cli import main
from synthts.edi import MU0, load_edi, site_from_edi

_EDI = """>HEAD
    DATAID="TEST01"
    LAT=30:30:00
    LONG=113:45:00
    ELEV=10.0
    EMPTY=1.0e32
>FREQ
  1.0 2.0 4.0 8.0
>ZXXR
  1 1 1 1
>ZXXI
  0 0 0 0
>ZXYR
  100 105 110 115
>ZXYI
  45 46 47 48
>ZYXR
  -100 -105 -110 -115
>ZYXI
  -45 -46 -47 -48
>ZYYR
  2 2 2 2
>ZYYI
  0.5 0.5 0.5 0.5
>TZXR
  0.1 0.1 0.1 0.1
>TZXI
  0.2 0.2 0.2 0.2
>TZYR
  0.3 0.3 0.3 0.3
>TZYI
  0.4 0.4 0.4 0.4
"""


@pytest.fixture
def edi_path(tmp_path):
    p = tmp_path / 'TEST01.EDI'
    p.write_text(_EDI, encoding='ascii')
    return str(p)


def test_load_edi_parse_and_units(edi_path):
    e = load_edi(edi_path)
    assert e['dataid'] == 'TEST01'
    assert e['lat'] == pytest.approx(30.5)
    assert e['lon'] == pytest.approx(113.75)
    assert list(e['freqs']) == [1.0, 2.0, 4.0, 8.0]
    fac = 1e-6 / (1e-9 / MU0)
    assert e['Z'][0, 1, 0] == pytest.approx(complex(100, 45) * fac)
    assert e['T'][1, 3] == pytest.approx(complex(0.3, 0.4))   # 倾子无量纲
    raw = load_edi(edi_path, to_si=False)
    assert raw['Z'][1, 0, 0] == pytest.approx(complex(-100, -45))


def test_site_from_edi_fingerprint(edi_path):
    e = load_edi(edi_path)
    s = site_from_edi(e, name='S1', densify=0)
    assert s.name == 'S1' and len(s.fields) == 4
    f0 = s.fields[0]
    assert f0.hx1 == 1 + 0j and f0.hy2 == 1 + 0j     # 单位极化基
    assert f0.ex2 == pytest.approx(e['Z'][0, 1, 0])   # Z@(0,1) → Ex=Zxy
    assert f0.ey1 == pytest.approx(e['Z'][1, 0, 0])
    assert f0.hz1 == pytest.approx(complex(0.1, 0.2))  # T·H
    assert f0.hz2 == pytest.approx(complex(0.3, 0.4))
    d = site_from_edi(e, name='S2', densify=2.0)
    assert len(d.fields) > 4                            # log 加密


def test_site_from_edi_rejects_barren(tmp_path):
    bad = _EDI.replace('ZXYR\n  100 105 110 115',
                       'ZXYR\n  1.0e32 1.0e32 1.0e32 1.0e32')
    p = tmp_path / 'BAD.EDI'
    p.write_text(bad, encoding='ascii')
    e = load_edi(str(p))
    with pytest.raises(ValueError, match='有效频点不足'):
        site_from_edi(e, densify=0)


def test_cli_edi_warning_and_summary(edi_path, tmp_path, capsys):
    out = str(tmp_path / 'sites')
    rc = main(['-edi', edi_path, '-out', out])
    assert rc == 0
    o = capsys.readouterr().out
    assert '非标准合成流程' in o and '循环论证' in o    # 强制警告横幅
    line = [ln for ln in o.splitlines()
            if ln.startswith('SYNTHTS_SUMMARY ')][-1]
    j = json.loads(line[len('SYNTHTS_SUMMARY '):])
    assert j['action'] == 'edi2xmlsite'
    assert 'warning' in j and '无偏真值' in j['warning']
    assert os.path.isfile(os.path.join(out, 'TEST01.XMLSITE'))
