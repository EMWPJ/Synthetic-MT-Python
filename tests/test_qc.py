# -*- coding: utf-8 -*-
"""QC 单测：小合成 → qc_directory 指标/JSON/图（matplotlib 可选）。"""
import json
import os

import numpy as np
import pytest

from synthts.cli import main
from synthts.model import Field, ForwardSite, save_site
from synthts.qc import discover_models, qc_directory, qc_site

RHO = 100.0


@pytest.fixture(scope='module')
def synth_out(tmp_path_factory):
    """半空间模型 + lines 引擎 1800s 合成（快、确定性）。"""
    d = tmp_path_factory.mktemp('qc_syn')
    site = ForwardSite()
    site.name = 'QC1'
    for f in np.logspace(-3, np.log10(320.0), 25):
        z = np.sqrt(2j * np.pi * f * 4e-7 * np.pi * RHO)
        fld = Field(frequency=float(f), is_noise=False)
        fld.ex1, fld.ey1 = 0.0j, -z
        fld.ex2, fld.ey2 = z, 0.0j
        fld.hx1, fld.hy1 = 1.0 + 0j, 0.0j
        fld.hx2, fld.hy2 = 0.0j, 1.0j
        site.fields.append(fld)
    model = d / 'qc1.XMLSITE'
    save_site(site, str(model))
    out = str(d / 'out').replace('\\', '/')
    rc = main(['-model', str(model), '-out', out, '-mode', 'MT',
               '-dur', '1800', '-seed', '42', '-source', 'lines'])
    assert rc == 0
    return out


def test_discover_models_dense_merge(synth_out):
    models = discover_models(synth_out)
    assert [n for n, _ in models] == ['QC1']     # _dense 归并，不重复


def test_qc_directory_metrics_and_json(synth_out):
    summary = qc_directory(synth_out, say=lambda *a: None)
    assert set(summary['sites']) == {'QC1'}
    m = summary['sites']['QC1']['metrics']
    assert m['XY']['n_freq'] >= 10
    assert m['XY']['rho_med_rel'] < 0.25         # lines 引擎锚点级精度
    assert m['YX']['ph_med_deg'] < 15.0
    j = json.load(open(os.path.join(synth_out, 'qc_summary.json'),
                       encoding='utf-8'))
    assert j['sites']['QC1']['metrics'] == m
    try:
        import matplotlib  # noqa: F401
        assert os.path.isfile(os.path.join(synth_out, 'qc_QC1.png'))
    except ImportError:
        assert not summary['png']


def test_qc_site_insufficient(tmp_path, synth_out):
    """空目录/无模型 → 明确报错。"""
    empty = tmp_path / 'nothing'
    empty.mkdir()
    with pytest.raises(ValueError, match='未发现'):
        qc_directory(str(empty), say=lambda *a: None)


def test_cli_qc_mode(synth_out, capsys):
    rc = main(['-qc', synth_out])
    assert rc == 0
    out = capsys.readouterr().out
    assert 'QC QC1' in out and 'qc_summary.json' in out
    assert out.rstrip().splitlines()[-1].startswith('SYNTHTS_SUMMARY ')
