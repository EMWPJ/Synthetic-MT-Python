# -*- coding: utf-8 -*-
"""golden 逐位回归：lines 引擎基线按 numpy 版本分桶（设计 §4.3）。

桶 = tests/golden/<numpy 版本>/hashes.json（纯数据文件，维护者生成）；
无桶版本自动跳过（统计等价由冒烟/闸门测试覆盖）。
桶再生（维护者，仓库根执行）：
  python tools/golden_regen.py    # 见该脚本（生成当前 numpy 版本的桶）
"""
import hashlib
import json
import os
import subprocess
import sys

import numpy as np
import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(REPO, 'examples', 'mta.XMLSITE')
EXTS = ('.TBL', '.TS3', '.TS4', '.TS5')
GOLDEN_ARGS = ['-model', MODEL, '-dur', '120', '-seed', '7', '-dense', '0',
               '-source', 'lines']   # 默认源 v2.0 起为 noise；golden 固定 lines


def _bucket_path():
    ver = np.__version__
    if not ver.replace('.', '').isdigit():
        pytest.skip('unexpected numpy version string')
    return os.path.join(REPO, 'tests', 'golden', ver, 'hashes.json')


def _current_hashes(tmp_path):
    out = tmp_path / 'g'
    r = subprocess.run(
        [sys.executable, '-m', 'synthts'] + GOLDEN_ARGS + ['-out', str(out)],
        cwd=str(tmp_path), capture_output=True, text=True,
        encoding='utf-8', errors='replace')
    assert r.returncode == 0, r.stdout + r.stderr
    files = {}
    for name in sorted(os.listdir(out)):
        if os.path.basename(name) == name and name.endswith(EXTS):
            with open(os.path.join(out, name), 'rb') as fh:
                files[name] = hashlib.sha256(fh.read()).hexdigest()
    return files


def test_golden_lines_byte_identical(tmp_path):
    path = _bucket_path()
    if not os.path.exists(path):
        pytest.skip('no golden bucket for numpy %s' % np.__version__)
    with open(path, encoding='utf-8') as fh:
        ref = json.load(fh)['files']
    assert _current_hashes(tmp_path) == ref, 'lines 引擎输出与 golden 基线不一致'
