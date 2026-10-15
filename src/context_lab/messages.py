"""消息模型 + token 估算器。

Msg 相比裸 dict 多一个 pinned 字段：关键决策点标记（如"客户确认
改地址"回合）。pinned 是给策略看的元数据，发送前由 to_api() 剥离
—— 协议层永远干净，元数据只在实验域内流转。

估算器是"决策用"而非"计费用"：预算分配器需要在发送前决定装多少
历史，这时只能估算；真实账单数字一律取 API 返回的 usage（llm.py），
两者在报告里分开呈现，不冒充。
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Msg:
    """策略层管理的历史消息。role: user / assistant / tool / system(标记)。"""

    role: str
    content: str = ""
    tool_calls: list[dict] | None = None  # assistant 发起的调用（openai 格式）
    tool_call_id: str | None = None
    pinned: bool = False  # 关键决策点：Budget 策略下不可压缩

    def to_api(self) -> dict:
        """转成 OpenAI 兼容协议的消息（剥离 pinned 等实验元数据）。"""
        if self.role == "tool":
            return {
                "role": "tool",
                "tool_call_id": self.tool_call_id or "",
                "content": self.content,
            }
        msg: dict = {"role": self.role, "content": self.content}
        if self.tool_calls:
            msg["tool_calls"] = self.tool_calls
        return msg


def estimate_tokens(text: str) -> int:
    """中英文混合的 token 估算：CJK ≈ 1 token/字，其余 ≈ 1/4 token/字符。"""
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return cjk + (other + 3) // 4


# 每条消息的协议开销（role 包装、分隔符等），与主流 tokenizer 行为同量级
PER_MESSAGE_OVERHEAD = 4


def estimate_messages_cost(msgs: list[Msg]) -> int:
    return sum(estimate_tokens(m.content) + PER_MESSAGE_OVERHEAD for m in msgs)


def estimate_api_messages_cost(api_msgs: list[dict]) -> int:
    """给假客户端用的同款估算（直接吃协议 dict）。"""
    total = 0
    for m in api_msgs:
        total += estimate_tokens(m.get("content") or "")
        total += estimate_tokens(str(m.get("tool_calls") or ""))
        total += PER_MESSAGE_OVERHEAD
    return total
