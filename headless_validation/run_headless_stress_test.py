"""Batch stress test for the dynamic BlueSky headless sector runner."""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "headless_dynamic_sector_validation.py"
LOG_DIR = ROOT / "headless_dynamic_logs"
SUMMARY_RE = re.compile(r"SUMMARY_PATH=(.+)")


def run_one(strategy: str, seed: int, timeout_sec: int) -> dict:
    env = os.environ.copy()
    env["ATC_RESOLUTION_PREFERENCE"] = strategy
    env["ATC_RNG_SEED"] = str(seed)
    proc = subprocess.run(
        [sys.executable, str(RUNNER)],
        cwd=str(ROOT),
        env=env,
        text=True,
        capture_output=True,
        timeout=timeout_sec,
    )
    summary_path = None
    for line in proc.stdout.splitlines():
        match = SUMMARY_RE.match(line.strip())
        if match:
            summary_path = Path(match.group(1))
    result = {
        "strategy": strategy,
        "seed": seed,
        "returncode": proc.returncode,
        "summary_path": str(summary_path) if summary_path else None,
        "stdout_tail": proc.stdout.splitlines()[-10:],
        "stderr_tail": proc.stderr.splitlines()[-10:],
    }
    if summary_path and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        result.update(
            {
                "success": bool(summary.get("success")),
                "num_loss_events": int(summary.get("num_loss_events", -1)),
                "num_commands": int(summary.get("num_commands", -1)),
                "fallback_calls": int(summary.get("fallback_calls", -1)),
                "min_hsep_nm": summary.get("min_hsep_nm"),
                "min_vsep_ft_when_hsep_lt_5nm": summary.get("min_vsep_ft_when_hsep_lt_5nm"),
                "log_path": summary.get("log_path"),
            }
        )
    else:
        result.update({"success": False, "num_loss_events": None, "num_commands": None, "fallback_calls": None})
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs-per-strategy", type=int, default=20)
    parser.add_argument("--base-seed", type=int, default=2026070300)
    parser.add_argument("--timeout-sec", type=int, default=180)
    parser.add_argument("--strategies", nargs="+", default=["altitude_first", "speed_first"])
    args = parser.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    aggregate_path = LOG_DIR / f"stress_test_{stamp}.json"
    results = []

    total = len(args.strategies) * args.runs_per_strategy
    completed = 0
    for strategy_idx, strategy in enumerate(args.strategies):
        for run_idx in range(args.runs_per_strategy):
            seed = args.base_seed + strategy_idx * 10000 + run_idx
            completed += 1
            print(f"[{completed}/{total}] strategy={strategy} seed={seed}", flush=True)
            try:
                result = run_one(strategy, seed, args.timeout_sec)
            except subprocess.TimeoutExpired as exc:
                result = {
                    "strategy": strategy,
                    "seed": seed,
                    "success": False,
                    "returncode": "timeout",
                    "num_loss_events": None,
                    "num_commands": None,
                    "fallback_calls": None,
                    "stdout_tail": (exc.stdout or "").splitlines()[-10:] if isinstance(exc.stdout, str) else [],
                    "stderr_tail": (exc.stderr or "").splitlines()[-10:] if isinstance(exc.stderr, str) else [],
                }
            results.append(result)
            print(
                "  "
                f"success={result.get('success')} "
                f"loss={result.get('num_loss_events')} "
                f"cmd={result.get('num_commands')} "
                f"fallback={result.get('fallback_calls')} "
                f"min_vsep={result.get('min_vsep_ft_when_hsep_lt_5nm')}",
                flush=True,
            )
            aggregate_path.write_text(json.dumps({"results": results}, indent=2, ensure_ascii=True), encoding="utf-8")

    by_strategy = {}
    for strategy in args.strategies:
        subset = [item for item in results if item["strategy"] == strategy]
        by_strategy[strategy] = {
            "runs": len(subset),
            "passed": sum(1 for item in subset if item.get("success") and item.get("num_loss_events") == 0),
            "failed": sum(1 for item in subset if not (item.get("success") and item.get("num_loss_events") == 0)),
            "total_loss_events": sum(int(item.get("num_loss_events") or 0) for item in subset),
            "max_fallback_calls": max([int(item.get("fallback_calls") or 0) for item in subset] or [0]),
            "avg_commands": sum(int(item.get("num_commands") or 0) for item in subset) / max(1, len(subset)),
        }
    payload = {"created_at": datetime.now().isoformat(timespec="seconds"), "summary": by_strategy, "results": results}
    aggregate_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"AGGREGATE_PATH={aggregate_path}", flush=True)
    print(json.dumps(by_strategy, indent=2, ensure_ascii=True), flush=True)
    return 0 if all(item.get("success") and item.get("num_loss_events") == 0 for item in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
