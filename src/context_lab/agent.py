"""带工具调用的多轮 agent 循环（实验的执行器）。
循环语义：user 进历史 -> 策略装配上下文 -> 调模型 -> 若发起工具调用，
执行后结果也进历史，重新装配再调 -> 文本回复进历史。
记账原则：每轮的 prompt_tokens 是"这轮所有 LLM 调用的输入 token 之和"
（含工具往返的中间调用）——长任务的真账单就是这么算的；管理开销
（策略的压缩调用）单独一列，不混进对话成本。
"""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Callable, Optional

from context_lab.llm import chat
from context_lab.messages import Msg
# ==================== 【修改1：导入工具截断函数】 ====================
from context_lab.strategies import ContextStrategy, truncate_tool_output

MAX_TOOL_ROUNDS = 4


@dataclass
class ToolSpec:
    """工具合同：schema 给模型看，fn 由本实验提供确定性实现。"""
    name: str
    description: str
    parameters: dict
    fn: Callable[..., str]

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class TurnRecord:
    turn: int
    user_text: str
    reply: str
    tool_calls: list[str] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    overhead_tokens: int = 0
    n_llm_calls: int = 0
    built_messages: int = 0  # 本轮实际装配进 prompt 的历史条数（可观测性）


def _execute_tool(tools: list[ToolSpec], call: dict) -> str:
    name = call["function"]["name"]
    for t in tools:
        if t.name == name:
            try:
                args = json.loads(call["function"]["arguments"] or "{}")
            except json.JSONDecodeError:
                return f"工具参数不是合法 JSON：{call['function']['arguments']!r}"
            try:
                return str(t.fn(**args))
            except Exception as err:  # noqa: BLE001 - 工具错误也走消息回填
                return f"工具执行出错：{err!r}"
    return f"未注册的工具：{name}"


def run_turn(
    *,
    strategy: ContextStrategy,
    system: str,
    tools: list[ToolSpec],
    user_text: str,
    turn: int,
    client: Optional[object] = None,
    pinned: bool = False,
) -> TurnRecord:
    """执行一个用户回合。pinned=True 时，本回合全部消息标记为关键决策点。"""
    overhead_before = strategy.overhead_tokens()
    strategy.add([Msg("user", user_text, pinned=pinned)])

    def api_messages() -> list[dict]:
        built = strategy.build()
        record_ref["built_messages"] = len(built)
        return [{"role": "system", "content": system}] + [m.to_api() for m in built]

    record_ref: dict = {"built_messages": 0}
    schemas = [t.schema() for t in tools] or None

    prompt = completion = calls = 0
    made_tools: list[str] = []
    reply_text = ""
    messages = api_messages()

    for _ in range(MAX_TOOL_ROUNDS + 1):
        result = chat(client=client, messages=messages, tools=schemas)
        prompt += result.prompt_tokens
        completion += result.completion_tokens
        calls += 1

        if result.tool_calls:
            strategy.add(
                [Msg("assistant", result.text, tool_calls=result.tool_calls, pinned=pinned)]
            )
            for call in result.tool_calls:
                made_tools.append(call["function"]["name"])
                # ==================== 【修改2：按策略开关截断工具结果】 ====================
                raw = _execute_tool(tools, call)
                if getattr(strategy, "tool_result_limit", None):
                    raw = truncate_tool_output(raw)

                strategy.add(
                    [
                        Msg(
                            "tool",
                            raw,
                            tool_call_id=call["id"],
                            pinned=pinned,
                        )
                    ]
                )
            messages = api_messages()
            continue

        strategy.add([Msg("assistant", result.text, pinned=pinned)])
        reply_text = result.text
        break
    else:
        reply_text = "（工具调用轮次超限，回合中止）"

    return TurnRecord(
        turn=turn,
        user_text=user_text,
        reply=reply_text,
        tool_calls=made_tools,
        prompt_tokens=prompt,
        completion_tokens=completion,
        overhead_tokens=strategy.overhead_tokens() - overhead_before,
        n_llm_calls=calls,
        built_messages=record_ref["built_messages"],
    )


def probe_answer(
    *,
    strategy: ContextStrategy,
    system: str,
    question: str,
    client: Optional[object] = None,
) -> tuple[str, int, int]:
    """期末测验：不带工具、只凭策略装配的上下文回答 —— 衡量"还记得多少"。"""
    probe_system = (
        f"{system}\n\n你现在回答期末测验题：只依据上下文里的已有信息作答，"
        "简短给出事实本身（日期/数字/名称），上下文里没有就说不知道。"
    )
    built = strategy.build()
    messages = [{"role": "system", "content": probe_system}]
    messages += [m.to_api() for m in built]
    messages += [{"role": "user", "content": question}]
    result = chat(client=client, messages=messages, tools=None, temperature=0.1)
    return result.text, result.prompt_tokens, result.completion_tokens
