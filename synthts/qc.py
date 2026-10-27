# -*- coding: utf-8 -*-
"""合成结果 QC：逐测点视电阻率/相位 理论 vs 合成数据 对比（图 + JSON）。

用法（CLI）：synthts -qc OUT_DIR
  模型发现：<site>_dense.XMLSITE 优先，回退 <site>.XMLSITE；
  阻抗估计：mtprocess.estimate_impedance_at（bypass_screen=True——
  噪声源半窗筛查通过率仅 ~4%，设计不适配 v2 噪声）；
  理论值  ：mtprocess.theoretical_impedance（E·H⁻¹，对任意指纹成立）；
  输出    ：qc_<site>.png（matplotlib 可选，缺省退化纯文本/JSON）+
            qc_summary.json + 文本表。

判读：噪声源在中低频有 CTS 最近邻×连续谱的物理地板（设计 §5.3），
中位 |Δρa/ρa| >10% 时给出提示行；不作为硬性失败（报告型工具）。
"""
import glob
import json
import math
import os

import numpy as np

MU0 = 4e-7 * math.pi


def _phi_deg(z):
    return math.degrees(math.atan2(z.imag, z.real))


def _fold_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def _safe_dir(data_dir):
    p = str(data_dir)
    if '..' in p.replace(chr(92), '/').split('/'):
        raise ValueError('目录路径不得包含 ..: %r' % p)
    return os.path.abspath(p)


def discover_models(data_dir):
    """[(site_name, model_path)]：*_dense.XMLSITE 归并为站名，dense 优先。"""
    data_dir = _safe_dir(data_dir)
    result = {}
    for p in glob.glob(os.path.join(data_dir, '*_dense.XMLSITE')):
        name = os.path.basename(p)[:-len('_dense.XMLSITE')]
        if name:
            result[name] = p
    for p in glob.glob(os.path.join(data_dir, '*.XMLSITE')):
        base = os.path.splitext(os.path.basename(p))[0]
        if base.endswith('_dense'):
            continue                     # 已按 dense 归并（或孤儿，跳过）
        result.setdefault(base, p)
    return sorted(result.items())


def qc_site(site_name, model_path, data_dir, max_freqs=40, make_png=True,
            verbose=False):
    """单站 QC → dict(freqs, rho_t/e, ph_t/e, metrics)。"""
    from . import mtprocess
    th = {round(f, 8): z for f, z in
          mtprocess.theoretical_impedance(model_path)}
    if not th:
        raise ValueError('模型无信号频点: %s' % model_path)
    freqs_all = np.array(sorted(th))
    k = min(max_freqs, len(freqs_all))
    sel = freqs_all[np.unique(np.linspace(0, len(freqs_all) - 1, k)
                              .round().astype(int))]
    rho_t = {'XY': [], 'YX': []}
    rho_e = {'XY': [], 'YX': []}
    ph_t = {'XY': [], 'YX': []}
    ph_e = {'XY': [], 'YX': []}
    used = []
    for f in sel:
        zth = th.get(round(float(f), 8))
        if zth is None or not np.isfinite(zth).all():
            continue
        r = mtprocess.estimate_impedance_at(float(f), data_dir, site_name,
                                            bypass_screen=True)
        if r is None:
            if verbose:
                print('  f=%.4g: 无覆盖频段，跳过' % f)
            continue
        zest, nwin = r
        if nwin < 1:
            continue
        used.append(float(f))
        for ij, (ri, ci) in (('XY', (0, 1)), ('YX', (1, 0))):
            w = 2 * math.pi * f
            zt, ze = zth[ri, ci], zest[ri, ci]
            rho_t[ij].append(abs(zt) ** 2 / (w * MU0))
            rho_e[ij].append(abs(ze) ** 2 / (w * MU0))
            ph_t[ij].append(_phi_deg(zt))
            ph_e[ij].append(_phi_deg(ze))
    if len(used) < 2:
        raise ValueError('%s: 可估计频点不足 2 个（数据太短或频段不覆盖）'
                         % site_name)
    metrics = {}
    for ij in ('XY', 'YX'):
        rt = np.array(rho_t[ij])
        re_ = np.array(rho_e[ij])
        rel = np.abs(re_ - rt) / np.abs(rt)
        dph = np.abs([_fold_deg(a - b)
                      for a, b in zip(ph_e[ij], ph_t[ij])])
        metrics[ij] = {'rho_med_rel': float(np.median(rel)),
                       'rho_max_rel': float(rel.max()),
                       'ph_med_deg': float(np.median(dph)),
                       'ph_max_deg': float(dph.max()),
                       'n_freq': int(len(rel))}
    return {'freqs': used, 'rho_t': rho_t, 'rho_e': rho_e,
            'ph_t': ph_t, 'ph_e': ph_e, 'metrics': metrics}


def _plot(site_name, qc, png_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f = np.array(qc['freqs'])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for ij, mk, cc in (('XY', 'o', 'tab:red'), ('YX', '^', 'tab:blue')):
        ax1.loglog(f, qc['rho_t'][ij], '-', color=cc, lw=1.2,
                   label='%s theory' % ij)
        ax1.loglog(f, qc['rho_e'][ij], mk, color=cc, ms=4, mfc='none',
                   label='%s synthetic' % ij)
        ax2.semilogx(f, qc['ph_t'][ij], '-', color=cc, lw=1.2)
        ax2.semilogx(f, qc['ph_e'][ij], mk, color=cc, ms=4, mfc='none')
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('Apparent resistivity (ohm·m)')
    ax1.legend(fontsize=8)
    ax1.grid(True, which='both', alpha=0.3)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('Phase (deg)')
    ax2.grid(True, which='both', alpha=0.3)
    fig.suptitle('%s — synthetic vs theory (%d freqs)' % (site_name,
                                                          len(f)))
    fig.tight_layout()
    fig.savefig(png_path, dpi=130)
    plt.close(fig)


def qc_directory(data_dir, max_freqs=40, make_png=True, say=print):
    """整个输出目录 QC：逐站图 + qc_summary.json；返回汇总 dict。"""
    data_dir = _safe_dir(data_dir)
    models = discover_models(data_dir)
    if not models:
        raise ValueError('目录中未发现 *.XMLSITE 模型（QC 需要合成输出'
                         '目录，含 <site>_dense.XMLSITE 与 TS 文件）: %r'
                         % data_dir)
    has_mpl = True
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        has_mpl = False
        say('  (matplotlib 未安装：跳过图片，仅输出文本/JSON —— '
            'pip install -e ".[plot]")')
    summary = {'data_dir': data_dir.replace('\\', '/'),
               'sites': {}, 'png': has_mpl and make_png}
    for name, path in models:
        say('  QC %s ...' % name)
        qc = qc_site(name, path, data_dir, max_freqs=max_freqs,
                     make_png=make_png, verbose=True)
        png = ''
        if make_png and has_mpl:
            png = os.path.join(data_dir, 'qc_%s.png' % name)
            _plot(name, qc, png)
            say('    图: %s' % png)
        m = qc['metrics']
        for ij in ('XY', 'YX'):
            say('    %s: ρa 中位 %.2f%% (max %.1f%%)  φ 中位 %.2f° '
                '(max %.1f°)  n=%d'
                % (ij, 100 * m[ij]['rho_med_rel'], 100 * m[ij]['rho_max_rel'],
                   m[ij]['ph_med_deg'], m[ij]['ph_max_deg'],
                   m[ij]['n_freq']))
        if max(m[ij]['rho_med_rel'] for ij in ('XY', 'YX')) > 0.10:
            say('    提示: 中位 ρa 偏差 >10% —— 噪声源低频有物理地板'
                '（CTS 最近邻 × 连续谱），短记录尤甚；加长 -dur 或参考'
                '设计文档 §5.3')
        summary['sites'][name] = {
            'model': os.path.basename(path),
            'metrics': m,
            'png': os.path.basename(png) if png else '',
            'freqs': qc['freqs'],
            'rho_theory': qc['rho_t'], 'rho_est': qc['rho_e'],
            'phase_theory': qc['ph_t'], 'phase_est': qc['ph_e'],
        }
    from pathlib import Path
    out = Path(data_dir) / 'qc_summary.json'
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=1),
                   encoding='utf-8')
    say('  汇总: %s（%d 站）' % (str(out).replace('\\', '/'), len(models)))
    return summary
