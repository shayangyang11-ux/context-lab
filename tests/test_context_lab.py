"""context-lab 测试：估算器、五策略、agent 循环、场景形状、端到端 dry-run。

纪律与 shop-agent 一致：假客户端注入生产代码路径，零网络零费用；
统计/策略行为用手算已知向量钉死。
"""

import json

import pytest

from context_lab.agent import probe_answer, run_turn
from context_lab.experiment import (
    _load_checkpoint,
    build_strategies,
    rule_compressor,
    run_experiment,
)
from context_lab.fakes import make_scenario_fake_client
from context_lab.messages import Msg, estimate_messages_cost, estimate_tokens
from context_lab.scenario import QUIZ, SYSTEM_PROMPT, build_scenario, check_answer
from context_lab.strategies import Budget, FullHistory, SlidingWindow, Summary, Truncation


# ------------------------------------------------------------- 估算器 --

def test_estimate_tokens_mixed():
    assert estimate_tokens("") == 0
    assert estimate_tokens("你好") == 2            # CJK ≈ 1/字
    assert estimate_tokens("abcd") == 1            # 4 ascii ≈ 1
    assert estimate_tokens("你好ab") == 3


def test_msg_to_api_strips_metadata():
    m = Msg("user", "hi", pinned=True)
    assert m.to_api() == {"role": "user", "content": "hi"}    # pinned 不进协议

    t = Msg("tool", "result", tool_call_id="call_1")
    api = t.to_api()
    assert api["role"] == "tool"
    assert api["tool_call_id"] == "call_1"

    a = Msg("assistant", "", tool_calls=[{"id": "c1", "type": "function",
                                          "function": {"name": "x", "arguments": "{}"}}])
    assert a.to_api()["tool_calls"][0]["function"]["name"] == "x"


def test_estimate_messages_cost_includes_overhead():
    msgs = [Msg("user", "你好"), Msg("assistant", "abcd")]
    # (2 + 4) + (1 + 4) = 11：估算值 = 内容 + 每条协议开销
    assert estimate_messages_cost(msgs) == 11


# --------------------------------------------------------------- 策略 --

def _fill(strategy, n, pin_idx=()):
    for i in range(n):
        strategy.add([Msg("user", f"消息{i}长长长长长", pinned=(i in pin_idx))])


def test_full_history_keeps_all():
    s = FullHistory()
    _fill(s, 12)
    assert len(s.build()) == 12
    assert s.overhead_tokens() == 0


def test_truncation_keeps_head_tail():
    s = Truncation(keep_first=2, keep_last=3)
    _fill(s, 10)
    built = s.build()
    assert len(built) == 6                       # 头2 + 标记 + 尾3
    assert built[0].content.startswith("消息0")
    assert built[-1].content.startswith("消息9")
    assert "5 条" in built[2].content            # 标记写明被截断的数量


def test_sliding_window_drops_old():
    s = SlidingWindow(max_messages=4)
    _fill(s, 10)
    built = s.build()
    assert len(built) == 5                       # 标记 + 最近4
    assert built[-1].content.startswith("消息9")
    assert built[1].content.startswith("消息6")   # 老的被丢


def test_summary_compresses_and_accounts_overhead():
    calls = []

    def fake_comp(evicted, prev):
        calls.append((len(evicted), prev))
        return "摘要内容", 42

    s = Summary(window=4, compressor=fake_comp)
    _fill(s, 10)                                 # 逐条 add：越窗即逐出
    assert sum(n for n, _ in calls) == 6         # 10 - 窗口4：逐出总量守恒
    assert s.overhead_tokens() == 42 * len(calls)  # 每次压缩成本都入账
    built = s.build()
    assert built[0].content.startswith("前情摘要")
    assert "摘要内容" in built[0].content
    assert len(built) == 5                       # 摘要 + 窗口4


def test_budget_keeps_pinned_over_budget():
    s = Budget(budget_tokens=30, compressor=lambda ev, prev: ("摘要", 5))
    _fill(s, 8, pin_idx=(2,))
    built = s.build()
    # 决策点恒保留：宁可顶破预算
    assert any("消息2" in m.content for m in built)
    assert s.pinned_overhead >= 0                # 溢出量被显式观测


def test_budget_generous_keeps_everything():
    s = Budget(budget_tokens=5000, compressor=lambda ev, prev: ("", 0))
    _fill(s, 6)
    assert len(s.build()) == 6                   # 预算够：不丢任何东西


def test_build_strategies_roster():
    strats = build_strategies(rule_compressor)
    names = [s.name for s in strats]
    assert names == ["full", "truncation", "sliding", "summary", "budget"]


# ------------------------------------------------------- agent 循环 --

def test_run_turn_tool_flow():
    client, calls = make_scenario_fake_client()
    s = FullHistory()
    tools, _ = build_scenario()
    rec = run_turn(
        strategy=s, system=SYSTEM_PROMPT, tools=tools,
        user_text="帮我看下订单A2026到哪了", turn=1, client=client,
    )
    assert rec.tool_calls == ["query_order"]     # 工具真的被调了
    assert rec.reply                              # 有最终文本
    assert rec.prompt_tokens > 0                  # usage 被记账
    assert rec.n_llm_calls >= 2                   # 发起调用 + 消化结果


def test_probe_answer_returns_text():
    client, _ = make_scenario_fake_client()
    s = FullHistory()
    tools, turns = build_scenario()
    run_turn(strategy=s, system=SYSTEM_PROMPT, tools=tools,
             user_text="你好", turn=1, client=client)
    text, p, c = probe_answer(strategy=s, system=SYSTEM_PROMPT,
                              question="礼物送给谁？", client=client)
    assert isinstance(text, str) and text
    assert p >= 0 and c >= 0


# ----------------------------------------------------------- 场景 --

def test_scenario_shape():
    tools, turns = build_scenario()
    names = {t.name for t in tools}
    assert {"query_order", "check_stock", "update_address"} <= names
    assert len(turns) == 25
    assert sum(1 for _, p in turns if p) == 2     # 两个 pinned 决策点
    assert len(QUIZ) == 6


def test_check_answer_normalization():
    assert check_answer("最晚10月28日前要到", ["10月28"])
    assert not check_answer("29号之前吧", ["10月28"])
    assert check_answer("预算300，后调到350", ["300", "350"])


# --------------------------------------------------------- 端到端 --

def test_experiment_dry_run(tmp_path):
    client, _ = make_scenario_fake_client()
    payload = run_experiment(client=client, out_dir=tmp_path)
    assert len(payload["results"]) == 5
    assert payload["n_turns"] == 25
    assert payload["dry_run"] is True
    for r in payload["results"]:
        assert len(r["records"]) == 25
        t = r["totals"]
        assert t["n_llm_calls"] >= 25
        assert t["prompt_tokens"] > 0
        assert t["quiz_total"] == 6
    # results.json 可反序列化，且包含结构化字段
    data = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert set(data["results"][0]["totals"]) >= {
        "prompt_tokens", "overhead_tokens", "quiz_hits", "max_turn_prompt"
    }


# ------------------------------------------------------- 断点续跑 --

def test_checkpoint_mode_mismatch_ignored(tmp_path, capsys):
    # 真跑遇到 dry-run 的旧 results.json：必须整档忽略，绝不拿假数据混账
    (tmp_path / "results.json").write_text(
        json.dumps({"dry_run": True, "results": [{"strategy": "full", "totals": {}}]}),
        encoding="utf-8",
    )
    assert _load_checkpoint(tmp_path, is_dry=False) == []
    assert "从头跑" in capsys.readouterr().out


def test_checkpoint_corrupt_file_ignored(tmp_path):
    # 写一半断电的残缺 JSON：当作没有断点，而不是崩溃
    (tmp_path / "results.json").write_text('{"dry_run": false, "resu', encoding="utf-8")
    assert _load_checkpoint(tmp_path, is_dry=False) == []


def test_experiment_resumes_from_checkpoint(tmp_path, capsys):
    client, _ = make_scenario_fake_client()
    run_experiment(client=client, out_dir=tmp_path)
    capsys.readouterr()  # 清掉第一场的输出
    first = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))

    # 模拟崩溃现场：4 条完整策略 + sliding 只跑到第 3 轮（partial）
    kept = [r for r in first["results"] if r["strategy"] != "sliding"]
    partial = {"strategy": "sliding", "records": kept[0]["records"][:3], "partial": True}
    (tmp_path / "results.json").write_text(
        json.dumps({**first, "results": kept + [partial]}, ensure_ascii=False),
        encoding="utf-8",
    )

    client2, _ = make_scenario_fake_client()
    payload = run_experiment(client=client2, out_dir=tmp_path)
    out = capsys.readouterr().out

    resume_line = out.split("[resume]")[1].split("\n")[0]
    assert "sliding" not in resume_line  # 半途而废的不算数，必须重跑
    assert "full" in resume_line and "budget" in resume_line
    assert out.count("[done]") == 1  # 只补跑 sliding 一条

    names = {r["strategy"] for r in payload["results"]}
    assert names == {"full", "truncation", "sliding", "summary", "budget"}
    again = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    by_name = {r["strategy"]: r for r in again["results"]}
    for r in kept:  # 断点里的完整策略原样复用，一个字节都不动
        assert by_name[r["strategy"]] == r
    assert len(by_name["sliding"]["records"]) == 25  # 重跑的是全新完整数据
    assert "totals" in by_name["sliding"]
    assert "partial" not in by_name["sliding"]

# ------------------------------------------------------- 工具结果截断 --
def test_truncate_tool_output_short_unchanged():
    """短文本小于阈值：原样返回，无截断标记，零开销"""
    from context_lab.strategies import truncate_tool_output, TOOL_RESULT_LIMIT
    short = "这是一段短工具返回，共100字符左右，远小于2000阈值。" * 5
    result = truncate_tool_output(short)
    assert result == short
    assert "……中间" not in result
    assert len(result) < TOOL_RESULT_LIMIT


def test_truncate_tool_output_long_head_tail():
    """长文本超过阈值：首尾保留、中段显式标记、总长度回落至可控范围"""
    from context_lab.strategies import truncate_tool_output, TOOL_RESULT_LIMIT
    # 构造远超阈值的长文本
    prefix = "【工具返回开头】订单查询结果："
    suffix = "【工具返回结尾】以上为全部数据。"
    middle = "字段内容填充行" * 600  # 约 6000 字符
    long_text = prefix + middle + suffix

    result = truncate_tool_output(long_text)
    # 首尾完整保留
    assert result.startswith(prefix)
    assert result.endswith(suffix)
    # 中段有截断标记
    assert "……中间" in result
    assert "字符已截断" in result
    assert "原文共" in result
    # 总长度控制在 2000 左右（900*2 + 标记文本 ≈ 1850）
    assert len(result) < 2200
    assert len(result) > 1700
    # 确认比原文短很多
    assert len(result) < len(long_text) / 3
