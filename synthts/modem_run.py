# -*- coding: utf-8 -*-
"""ModEM 全流程一键编排：三维模型 + 常规 dat → 合成测点（XMLSITE）。

流程（SKILL 任务 8）：
  1. 解析 dat 的 (周期, 测站) 清单，按 -stations/-max-periods 取子集；
  2. add_em_fields 补 EM_Fields 占位段（synthts.modem）；
  3. 模型/数据拷入**短 ASCII 工作目录**再调改版 exe —— exe 命令行缓冲
     装不下长中文路径（参数截断 → No input model parametrization，实测
     ID005 中文路径必败、同文件 ASCII 拷贝成功）；
  4. `-F model prepped.dat resp.dat fields.E par` 正演，回填场值；
  5. 响应 → XMLSITE 测点（sites_from_response，SI/直通口径）。

响应选择：exe 把结果写进第 3 个命令行参数（resp.dat）；个别版本会原位
更新输入文件——resp 无回填时回退检查 prepped。fields.E（全网格解，
数百 MB）默认随工作目录删除，-keep-work 保留。
"""
import os
import re
import shutil
import subprocess
import tempfile

from .modem import (add_em_fields, parse_modem_response,
                    sites_from_response, write_modem_sites)

# 缺省正演参数（COMMEMI 例子同款，紧容差）
_DEFAULT_PAR = (
    'Number of QMR iters per divergence correction : 100\n'
    'Maximum number of divergence correction calls : 50\n'
    'Maximum number of divergence correction iters : 200\n'
    'Misfit tolerance for EM forward solver        : 1.0e-12\n'
    'Misfit tolerance for EM adjoint solver        : 1.0e-12\n'
    'Misfit tolerance for divergence correction    : 1.0e-10\n')

_COUNTS_RE = re.compile(r'^\s*>\s*\d+\s+\d+\s*$')

_WORK_PATH_LIMIT = 60      # exe 命令行参数安全长度上限（留裕量）


def _safe_check(path, what):
    p = str(path)
    if '..' in p.replace(chr(92), '/').split('/'):
        raise ValueError('%s 路径不得包含 ..: %r' % (what, p))
    return p


def logspaced_idx(n, k):
    """[0,n) 中 k 个等距指标（含首尾，去重保序）。"""
    if k >= n:
        return list(range(n))
    if k < 2:
        return [0]
    pos = {int(round(i * (n - 1) / (k - 1))) for i in range(k)}
    return sorted(pos)


def subset_dat(src, dst, stations=None, max_periods=None):
    """按测站码/周期数取子集：保持段结构与行序，重写各段 counts 行。

    stations: 保留的站码集合；max_periods: 保留周期个数（等距取，含首尾）。
    counts = (周期数, 唯一站位数)——与 add_em_fields 同口径。
    返回 dict(n_periods, n_stations, rows)。
    """
    _safe_check(src, 'modem 输入')
    _safe_check(dst, 'modem 输出')
    from pathlib import Path
    lines = Path(src).read_bytes().decode(
        'utf-8', errors='replace').splitlines(True)
    resp = parse_modem_response(src)
    if not resp.record_order:
        raise ValueError('ModEM dat 无数据行: %s' % src)
    keep_stations = set(stations) if stations else None
    keep_periods = resp.periods
    if max_periods and max_periods < len(resp.periods):
        keep_periods = [resp.periods[i]
                        for i in logspaced_idx(len(resp.periods), max_periods)]

    def row_of(ln):
        """数据行 → (per, code)；非数据行 → None。"""
        t = ln.split()
        if len(t) < 8:
            return None
        try:
            per = float(t[0])
        except ValueError:
            return None
        return per, t[1]

    def keep(per, code):
        if keep_stations is not None and code not in keep_stations:
            return False
        return any(abs(per - p) <= 1e-9 * max(1.0, abs(p))
                   for p in keep_periods)

    final = []
    # counts 行在段数据行**之前**：段 k = 第 k 个 counts 行之后到下一
    # counts 行（或文件尾）的数据行。两遍法：段号 = 已见 counts 行数。
    seg_stats = {}                      # 段号 -> [per_set, sta_set]
    n_counts = 0
    for ln in lines:
        if _COUNTS_RE.match(ln):
            n_counts += 1
            continue
        row = row_of(ln)
        if row is not None and keep(*row):
            st = seg_stats.setdefault(n_counts, [set(), set()])
            st[0].add(row[0])
            st[1].add(row[1])
    n_counts = 0
    n_per, n_sta, n_rows = set(), set(), 0
    for ln in lines:
        if _COUNTS_RE.match(ln):
            n_counts += 1
            per, sta = seg_stats.get(n_counts, [set(), set()])
            final.append('>%8d%6d\r\n' % (len(per), len(sta)))
            n_per.update(per)
            n_sta.update(sta)
            continue
        row = row_of(ln)
        if row is None:
            final.append(ln)            # 注释/段头/空行原样保留
        elif keep(*row):
            final.append(ln)
            n_rows += 1
        # else: 未选中的数据行剔除
    Path(dst).write_bytes(''.join(final).encode('utf-8'))
    return {'n_periods': len(n_per), 'n_stations': len(n_sta),
            'rows': n_rows}


def _work_dir():
    """短 ASCII 工作目录（规避 exe 命令行缓冲限制）。"""
    base = os.environ.get('SYNTHTS_MODEM_WORK') or tempfile.gettempdir()
    if not base.isascii() or len(base) > _WORK_PATH_LIMIT - 20:
        raise ValueError(
            '临时目录不适合 ModEM exe（需 ASCII 且较短）: %r\n'
            '  请设置 SYNTHTS_MODEM_WORK 指向短 ASCII 目录'
            '（如 D:\\modem_work）' % base)
    d = tempfile.mkdtemp(prefix='st_modem_', dir=base)
    if len(d) > _WORK_PATH_LIMIT:
        raise ValueError('工作目录过长（exe 命令行会截断）: %r\n'
                         '  请设置 SYNTHTS_MODEM_WORK' % d)
    return d


def run_forward(exe, model, dat, resp_path, e_path, par, log_path):
    """调改版 exe `-F model dat resp E par`；返回 (rc, 日志尾部)。"""
    _safe_check(exe, 'ModEM exe')
    cmd = [os.path.abspath(exe), '-F', model, dat, resp_path, e_path, par]
    proc = subprocess.run(cmd, cwd=os.path.dirname(model) or '.',
                          capture_output=True, text=True, errors='replace')
    from pathlib import Path
    Path(log_path).write_text((proc.stdout or '') + (proc.stderr or ''),
                              encoding='utf-8', errors='replace')
    tail = ((proc.stderr or '') + (proc.stdout or ''))[-1500:]
    return proc.returncode, tail


def orchestrate(exe, model_path, dat_path, out_dir, stations=None,
                max_periods=None, keep_work=False, par_file=None, say=print):
    """一键流程：子集 → 补段 → exe 正演 → XMLSITE 测点。

    返回 (site_paths, meta)。exe 不存在/失败/无回填时抛 ValueError。
    """
    _safe_check(model_path, '模型')
    _safe_check(dat_path, '数据')
    from pathlib import Path
    resp0 = parse_modem_response(dat_path)
    if resp0.has_em_fields:
        raise ValueError('dat 已含 EM_Fields 段（应传入常规 ModEM 数据'
                         '文件）: %s' % dat_path)
    if not os.path.isfile(exe):
        raise ValueError('ModEM exe 不存在: %r（用 -modem-exe 指定改版 exe）'
                         % exe)
    os.makedirs(out_dir, exist_ok=True)
    fwd_dir = os.path.join(out_dir, 'fwd')
    os.makedirs(fwd_dir, exist_ok=True)
    work = _work_dir()
    try:
        say('  subset: 源 %d 周期 × %d 站'
            % (len(resp0.periods), len(resp0.station_keys())))
        sub = os.path.join(work, 'd.dat')
        info = subset_dat(dat_path, sub, stations=stations,
                          max_periods=max_periods)
        say('  subset→ %d 周期 × %d 站（%d 行）'
            % (info['n_periods'], info['n_stations'], info['rows']))
        prepped = os.path.join(work, 'p.dat')
        add_em_fields(sub, prepped)
        say('  fields: 已补 EM_Fields 占位段（初值 0，正演回填）')
        m_file = os.path.join(work, 'm.modem')
        shutil.copyfile(model_path, m_file)
        par = os.path.join(work, 'f.par')
        if par_file:
            shutil.copyfile(par_file, par)
        else:
            Path(par).write_text(_DEFAULT_PAR, encoding='ascii')
        resp_path = os.path.join(work, 'r.dat')
        e_path = os.path.join(work, 'f.E')
        log_path = os.path.join(fwd_dir, 'modem_fwd.log')
        say('  exe   : %s -F m.modem p.dat r.dat f.E f.par（work=%s）'
            % (os.path.basename(exe), work))
        rc, tail = run_forward(exe, m_file, prepped, resp_path, e_path, par,
                               log_path)
        if rc != 0:
            raise ValueError('ModEM exe 失败 rc=%d；日志尾部:\n%s'
                             % (rc, tail))
        r = parse_modem_response(resp_path)
        used = resp_path
        if not (r.fields_present and r.fields_nonzero):
            r = parse_modem_response(prepped)     # 个别版本原位更新输入
            used = prepped
        if not (r.fields_present and r.fields_nonzero):
            raise ValueError('正演输出无回填的 EM_Fields 场值；日志尾部:\n%s'
                             % tail)
        say('  filled: 场值已回填（来源 %s）' % os.path.basename(used))
        sites = sites_from_response(r)
        paths = write_modem_sites(sites, out_dir)
        for p in paths:
            say('  site  : %s' % p)
        shutil.copyfile(used, os.path.join(fwd_dir, 'response.dat'))
        meta = {'n_periods': info['n_periods'],
                'n_stations': info['n_stations'],
                'n_sites': len(sites),
                'work_dir': work if keep_work else '',
                'work_kept': bool(keep_work)}
        return paths, meta
    finally:
        if keep_work:
            say('  work  : 保留 %s' % work)
        else:
            shutil.rmtree(work, ignore_errors=True)
