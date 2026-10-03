"""CLI 冒烟测试：cli.py run / demo 端到端可执行且返回约定退出码。

作者：晨星
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CLI = str(REPO / "cli.py")


def _run(args):
    # check=False：由各用例自行断言 returncode，避免 subprocess 抛异常掩盖真实输出
    return subprocess.run(
        [sys.executable, CLI, *args],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


def test_cli_run_smoke():
    r = _run(["run", "--seed", "0"])
    assert r.returncode == 0, r.stderr
    assert "overfit_gap" in r.stdout


def test_cli_demo_determinism_and_gate():
    out = str(REPO / "examples" / "benchmark_ci.json")
    r = _run(["demo", "--out", out])
    # demo: 0=通过(门禁+确定性) 1=门禁未过 2=确定性失败
    assert r.returncode == 0, r.stderr
    rep = json.loads(Path(out).read_text(encoding="utf-8"))
    assert rep["gate"]["passed"] is True
