"""端到端演示：多 seed 基准 → 落盘 benchmark.json → 二次运行确定性自检。

作者：晨星
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.config import MiaConfig
from pipeline.pipeline import benchmark


def _dump(report: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)


def _core_numbers(report: dict) -> dict:
    """用于确定性对比的核心数值（排除非确定性字段）。"""
    return {
        "aggregate": report["aggregate"],
        "gate": report["gate"],
        "ablation": report["ablation"],
        "per_seed": report["per_seed"],
    }


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="examples/benchmark.json")
    ap.add_argument("--n-seeds", dest="n_seeds", type=int, default=None)
    ap.add_argument("--snr", type=float, default=None)
    ap.add_argument("--n-train", dest="n_train", type=int, default=None)
    args = ap.parse_args(argv)

    cfg = MiaConfig()
    if args.n_seeds is not None:
        cfg.n_seeds = args.n_seeds
    if args.snr is not None:
        cfg.snr = args.snr
    if args.n_train is not None:
        cfg.n_train = args.n_train
        cfg.n_nonmember = args.n_train
    cfg.validate()

    print("[demo] run #1 ...")
    t0 = time.time()
    r1 = benchmark(cfg)
    d1 = time.time() - t0

    _dump(r1, args.out)

    print("[demo] run #2 (determinism check) ...")
    r2 = benchmark(cfg)
    c1 = _core_numbers(r1)
    c2 = _core_numbers(r2)
    bit_identical = json.dumps(c1, sort_keys=True) == json.dumps(c2, sort_keys=True)

    # —— 汇总打印 ——
    print("\n=== MiaForge end-to-end demo ===")
    print(f"elapsed (#1): {d1:.2f}s | determinism bit-identical: {bit_identical}")
    print(f"{'method':<22}{'AUC mean':>10}{'AUC std':>10}")
    for name, a in r1["aggregate"].items():
        print(f"{name:<22}{a['mean']:>10.4f}{a['std']:>10.4f}")
    g = r1["gate"]
    print(f"\nGATE delta={g['delta']} (>= {g['gate_delta_required']}) significant={g['significant']} PASSED={g['passed']}")
    print(f"\nfailures collected: {len(r1['failures'])}")
    print(f"wrote {args.out}")

    if not bit_identical:
        print("[ERROR] determinism FAILED: two runs differ", file=sys.stderr)
        return 2
    if not g["passed"]:
        print("[WARN] gate NOT passed (see report)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
