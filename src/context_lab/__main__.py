"""CLI 入口。

用法：
    python -m context_lab --dry-run                # 假客户端零费用验证管线
    python -m context_lab                          # 真实 LLM 实验（读 .env）
    python -m context_lab --report <results.json>  # 只重新生成报告与图表
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.console import Console

console = Console()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m context_lab")
    parser.add_argument("--dry-run", action="store_true", help="假客户端零费用验证管线")
    parser.add_argument("--out", default="context_lab_results", help="产物目录")
    parser.add_argument("--report", default=None, help="从 results.json 重新生成报告")
    args = parser.parse_args(argv)

    out = Path(args.out)

    if args.report:
        from context_lab.report import write_outputs

        payload = json.loads(Path(args.report).read_text(encoding="utf-8"))
        for p in write_outputs(payload, out):
            console.print(f"已写入：{p}")
        return 0

    client = None
    if args.dry_run:
        from context_lab.fakes import make_scenario_fake_client

        client, _ = make_scenario_fake_client()
        console.print("[yellow]DRY-RUN：场景感知假客户端 + 规则压缩器，数字仅证明管线形状[/yellow]")

    from context_lab.experiment import run_experiment
    from context_lab.report import write_outputs

    payload = run_experiment(client=client, out_dir=out)
    for p in write_outputs(payload, out):
        console.print(f"已写入：{p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
