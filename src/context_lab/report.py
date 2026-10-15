"""结果呈现：results.json -> report.md + tokens.png（缺 matplotlib 时自动降级）。"""
from __future__ import annotations
from pathlib import Path

# ==================== 【修改1：策略标签重命名，明确分层定位】 ====================
CN = {
    "full": "全量保留（基线）",
    "truncation": "截断（消息级·对照组）",
    "sliding": "滑窗（最近8条）",
    "summary": "三层堆叠（工具截断+滑窗8+摘要）",
    "budget": "三层+预算分配",
}


def _cn(name: str) -> str:
    return CN.get(name, name)


def write_outputs(payload: dict, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = [write_report(payload, out)]
    chart = write_chart(payload, out)
    if chart:
        paths.append(chart)
    return paths


def write_report(payload: dict, out_dir: Path) -> Path:
    # 断点文件可能混有未完成的策略（partial 且无 totals）：报告只认完整的
    rs = [r for r in payload["results"] if not r.get("partial") and "totals" in r]
    by = {r["strategy"]: r for r in rs}
    base = by.get("full", {}).get("totals", {}).get("prompt_tokens") or 0
    # ==================== 【新增：计算总成本基线】 ====================
    base_total = 0
    if "full" in by:
        ft = by["full"]["totals"]
        base_total = ft["prompt_tokens"] + ft["completion_tokens"] + ft["overhead_tokens"]

    lines = [
        "# context-lab：25 轮长任务上下文管理对比实验",
        "",
        "## 一、实验设置",
        "",
        f"- 场景：「山野杂货铺」客服长任务，{payload['n_turns']} 轮 + 6 题期末测验",
        f"- 第 0 层固定成本（system + 工具 schema）：约 "
        f"{payload['system_layer_tokens_per_turn']} token/轮，已含在各策略 prompt_tokens 内"
        " —— 这是所有策略都省不掉的地板",
        "- 记账口径：输入/输出为 API 返回的真实 usage；管理开销（overhead）"
        "为策略压缩调用单独花费；期末测验成本单独列出，不混账",
    ]
    if payload.get("dry_run"):
        lines.append("- ⚠️ 本报告由 --dry-run 生成（假客户端 + 规则压缩器）："
                     "数字只证明管线形状正确，不是实验结论")
    lines += [
        "",
        "## 二、总账",
        "",
        # ==================== 【修改2：总账表新增总成本列与基线对比】 ====================
        "| 策略 | 输入token | 较基线输入 | 单轮峰值 | 输出token | 管理开销 | 总成本 | 较基线总成本 | 工具调用 | 测验命中 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rs:
        t = r["totals"]
        pct = f"{(1 - t['prompt_tokens'] / base):.0%}" if base else "-"
        # 计算全口径总成本 + 基线对比
        total_cost = t["prompt_tokens"] + t["completion_tokens"] + t["overhead_tokens"]
        pct_total = f"{(1 - total_cost / base_total):.0%}" if base_total else "-"
        lines.append(
            f"| {_cn(r['strategy'])} | {t['prompt_tokens']:,} | {pct} "
            f"| {t['max_turn_prompt']:,} | {t['completion_tokens']:,} "
            f"| {t['overhead_tokens']:,} | {total_cost:,} | {pct_total} "
            f"| {t['n_tool_calls']} | {t['quiz_hits']}/{t['quiz_total']} |"
        )
    lines += [
        "",
        "## 三、期末测验明细（记忆保持）",
        "",
        "| 策略 | " + " | ".join(f"Q{i}" for i in range(1, len(rs[0]["quiz"]) + 1)) + " |",
        "|---|" + "---|" * len(rs[0]["quiz"]),
    ]
    for r in rs:
        marks = " | ".join("✓" if q["hit"] else "✗" for q in r["quiz"])
        lines.append(f"| {_cn(r['strategy'])} | {marks} |")
    lines += ["", "题面与关键事实：", ""]
    for i, q in enumerate(rs[0]["quiz"], 1):
        lines.append(f"{i}. [{q['depth']}] {q['q']}（关键词：{'、'.join(q['keys'])}）")
    lines += ["", "## 四、自动观察", ""]
    if rs:
        cheapest = min(rs, key=lambda r: r["totals"]["prompt_tokens"])
        best_mem = max(rs, key=lambda r: r["totals"]["quiz_hits"])
        # ==================== 【修改3：新增总成本最低项，呈现反直觉结论】 ====================
        cheapest_total = min(
            rs,
            key=lambda r: r["totals"]["prompt_tokens"] + r["totals"]["completion_tokens"] + r["totals"]["overhead_tokens"]
        )
        lines.append(
            f"- 最省输入：{_cn(cheapest['strategy'])}"
            f"（{cheapest['totals']['prompt_tokens']:,} token）"
        )
        lines.append(
            f"- 总成本最低：{_cn(cheapest_total['strategy'])}"
            f"（{cheapest_total['totals']['prompt_tokens'] + cheapest_total['totals']['completion_tokens'] + cheapest_total['totals']['overhead_tokens']:,} token）"
        )
        lines.append(
            f"- 记忆最好：{_cn(best_mem['strategy'])}"
            f"（{best_mem['totals']['quiz_hits']}/{best_mem['totals']['quiz_total']}）"
        )
        for r in rs:
            t = r["totals"]
            if t["overhead_tokens"] and base:
                saved = base - t["prompt_tokens"]
                verdict = "回本" if saved > t["overhead_tokens"] else "未回本"
                lines.append(
                    f"- {_cn(r['strategy'])}：压缩管理花费 {t['overhead_tokens']:,}，"
                    f"换回输入节省 {saved:,}（{verdict}）"
                )
    lines += [
        "",
        "## 五、分析框架（读数指南）",
        "",
        "- 全量基线的单轮峰值随轮数近似线性增长 —— 长任务下时延与费用双杀；",
        "- 滑窗最便宜，但早期事实直接丢（测验早期题应明显丢分）—— 省钱丢记忆；",
        "- 截断保头保尾：开场约束还在，但中段事实（预算/期限）丢失；",
        "- 摘要/预算用管理开销买记忆：看『节省是否覆盖开销』与『测验命中』两栏；",
        "- 预算策略的 pinned 例外：关键决策点顶破预算也保留"
        "（Budget.pinned_overhead 字段观测这一代价）。",
        "- 总成本口径下，压缩策略可能从「最省」翻转为「最贵」——管理开销是隐形账单。",
        "",
        "## 六、复现",
        "",
        "```bash",
        "python -m context_lab                                   # 真实 LLM",
        "python -m context_lab --report context_lab_results/results.json",
        "```",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_chart(payload: dict, out_dir: Path) -> Path | None:
    """双面板图：左=每轮输入（看策略是否压平增长），右=累计输入（看总账）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
    for r in payload["results"]:
        per_turn = [rec["prompt_tokens"] for rec in r["records"]]
        cum, acc = [], 0
        for v in per_turn:
            acc += v
            cum.append(acc)
        x = range(1, len(per_turn) + 1)
        ax1.plot(x, per_turn, marker="o", markersize=3, label=r["strategy"])
        ax2.plot(x, cum, marker="o", markersize=3, label=r["strategy"])
    ax1.set_title("Input tokens per turn")
    ax1.set_xlabel("Turn")
    ax1.set_ylabel("prompt tokens")
    ax2.set_title("Cumulative input tokens")
    ax2.set_xlabel("Turn")
    for ax in (ax1, ax2):
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "tokens.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
