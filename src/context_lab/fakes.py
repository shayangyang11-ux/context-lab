"""场景感知假客户端：--dry-run 与测试共用，零费用验证整条实验管线。

与"队列脚本式"假客户端不同，它读最后一条消息的内容决定行为
（回文本 / 发起工具调用），usage 由估算器现场计算 —— dry-run 的
token 数字是"管线形状正确"的证明，不是实验结论。
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

from context_lab.messages import estimate_api_messages_cost, estimate_tokens


def make_scenario_fake_client():
    """返回 (假客户端, 调用记录)。"""

    calls: list[dict] = []

    def _usage(messages: list[dict], content: str):
        return SimpleNamespace(
            prompt_tokens=estimate_api_messages_cost(messages),
            completion_tokens=estimate_tokens(content) + 8,
        )

    def _text(messages, content):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=None))],
            usage=_usage(messages, content),
        )

    def _tool_call(messages, name: str, arguments: dict):
        call = {
            "id": f"call_{len(calls) + 1}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(arguments, ensure_ascii=False)},
        }
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=None, tool_calls=[SimpleNamespace(
                        id=call["id"],
                        function=SimpleNamespace(
                            name=call["function"]["name"],
                            arguments=call["function"]["arguments"],
                        ),
                    )])
                )
            ],
            usage=_usage(messages, ""),
        )

    def create(**kwargs):
        calls.append(kwargs)
        messages: list[dict] = kwargs["messages"]
        has_tools = bool(kwargs.get("tools"))
        last = messages[-1]
        last_text = last.get("content") or ""

        if last["role"] == "tool":
            return _text(messages, f"查到了：{last_text[:80]}。还有什么可以帮您？")

        # 测验探针（无工具、问题式）
        if not has_tools and ("？" in last_text or "?" in last_text):
            return _text(messages, "（dry-run 模拟回答：不含真实事实）")

        if has_tools and re.search(r"(到哪|物流|状态|订单)", last_text) and "A2026" in last_text:
            return _tool_call(messages, "query_order", {"order_id": "A2026"})

        if has_tools and re.search(r"(有货|库存|多少钱|还能买|还有吗|微波炉)", last_text):
            if "手账" in last_text:
                return _tool_call(messages, "check_stock", {"sku": "JOURNAL-01"})
            return _tool_call(messages, "check_stock", {"sku": "MUG-01"})

        if (
            has_tools
            and re.search(r"(改寄|改到|修改.*地址|寄到上海)", last_text)
            and "确认" in last_text
        ):
            return _tool_call(
                messages,
                "update_address",
                {"order_id": "A2026", "new_address": "上海市浦东新区"},
            )

        if "总结" in last_text:
            return _text(
                messages,
                "本次对话要点：礼物对象与偏好、订单 A2026、预算与期限、地址变更（dry-run 模拟）。",
            )

        return _text(messages, "好的，已经为您记录。还需要别的吗？（dry-run 模拟回复）")

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    return client, calls
