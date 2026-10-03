"""MiaForge 命令行入口。

用法：
  python cli.py benchmark --out examples/benchmark.json
  python cli.py run --seed 0
  python cli.py demo        # 等同 benchmark（端到端演示 + 确定性自检）

作者：晨星
"""

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import MiaConfig
from pipeline.pipeline import benchmark, run_one


def _print_summary(report: dict) -> None:
    print("\n=== MiaForge benchmark summary ===")
    cfg = report["config"]
    print(f"config: n_train={cfg['n_train']} n_nonmember={cfg['n_nonmember']} "
          f"n_features={cfg['n_features']} snr={cfg['snr']} hidden={cfg['hidden']} "
          f"n_seeds={cfg['n_seeds']}")
    print(f"{'method':<22}{'AUC mean':>10}{'AUC std':>10}")
    for name, a in report["aggregate"].items():
        print(f"{name:<22}{a['mean']:>10.4f}{a['std']:>10.4f}")
    g = report["gate"]
    print(f"\nGATE baseline={g['baseline']} AUC={g['baseline_auc_mean']} | "
          f"flagship={g['flagship']} AUC={g['flagship_auc_mean']}")
    print(f"delta={g['delta']} (required >= {g['gate_delta_required']}) | "
          f"significant={g['significant']} | PASSED={g['passed']}")
    print("\nAblation (vs flagship):")
    for name, a in report["ablation"].items():
        print(f"  {name:<22} AUC={a['auc_mean']:.4f} delta={a['delta_vs_flagship']:+.4f}")
    print(f"\nFailure cases collected: {len(report['failures'])}")


def cmd_benchmark(args) -> int:
    cfg = MiaConfig.from_env() if args.env else MiaConfig()
    if args.seed is not None:
        cfg.base_seed = args.seed
    if args.n_train is not None:
        cfg.n_train = args.n_train
    if args.n_nonmember is not None:
        cfg.n_nonmember = args.n_nonmember
    if args.snr is not None:
        cfg.snr = args.snr
    if args.n_seeds is not None:
        cfg.n_seeds = args.n_seeds
    cfg.validate()
    report = benchmark(cfg)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"wrote {args.out}")
    _print_summary(report)
    return 0 if report["gate"]["passed"] else 1


def cmd_run(args) -> int:
    cfg = MiaConfig.from_env() if args.env else MiaConfig()
    rec = run_one(cfg, args.seed)
    print(f"seed={rec.seed} target_train_loss={rec.target_train_loss:.4f} "
          f"target_eval_loss={rec.target_eval_loss:.4f} overfit_gap={rec.overfit_gap:.4f}")
    print(f"{'method':<22}{'AUC':>10}{'acc':>10}")
    for name in rec.methods:
        print(f"{name:<22}{rec.methods[name]:>10.4f}{rec.accuracy[name]:>10.4f}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="MiaForge — Membership Inference Attack system")
    sub = ap.add_subparsers(dest="cmd")

    b = sub.add_parser("benchmark", help="run multi-seed benchmark")
    b.add_argument("--out", default=None)
    b.add_argument("--seed", type=int, default=None)
    b.add_argument("--n-train", dest="n_train", type=int, default=None)
    b.add_argument("--n-nonmember", dest="n_nonmember", type=int, default=None)
    b.add_argument("--snr", type=float, default=None)
    b.add_argument("--n-seeds", dest="n_seeds", type=int, default=None)
    b.add_argument("--env", action="store_true", help="apply ENV_MIA_* overrides")

    r = sub.add_parser("run", help="run single seed")
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--env", action="store_true")

    d = sub.add_parser("demo", help="benchmark + determinism self-check")
    d.add_argument("--out", default="examples/benchmark.json")
    d.add_argument("--n-seeds", dest="n_seeds", type=int, default=None)

    args = ap.parse_args(argv)
    if args.cmd == "benchmark":
        return cmd_benchmark(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "demo":
        # 复用 benchmark 并把确定性自检交给 run_demo
        import examples.run_demo as demo
        return demo.main(["--out", args.out] + (["--n-seeds", str(args.n_seeds)] if args.n_seeds else []))
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
