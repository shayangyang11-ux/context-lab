"""LLM 基座：OpenAI 兼容 chat + 工具调用 + 真实 usage 记账。

与 shop-agent 的 llm.py 同一工程范式（依赖注入、延迟 import、
client 可被测试替身替换），但职责不同：这里要的是"带工具调用的
原始 chat + usage 数字"，不是结构化输出 —— 上下文实验关心的是
token 账单，每次调用的 prompt_tokens / completion_tokens 都是
实验的原始数据。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from context_lab.config import get_settings


@dataclass
class ChatReply:
    text: str = ""
    tool_calls: list[dict] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0


def get_client():
    """延迟创建真实客户端（模块导入零开销；测试/干跑永不走到这）。"""
    from openai import OpenAI

    settings = get_settings()
    # 下限保护：批处理实验宁可单次多等几分钟，也不要 60s×3 连环超时把整场打死
    return OpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        timeout=max(180.0, settings.llm_timeout),
    )


def _normalize_tool_calls(raw) -> list[dict]:
    if not raw:
        return []
    calls = []
    for tc in raw:
        fn = tc.function
        calls.append(
            {
                "id": tc.id,
                "type": "function",
                "function": {"name": fn.name, "arguments": fn.arguments},
            }
        )
    return calls


def chat(
    *,
    client: Optional[object] = None,
    messages: list[dict],
    tools: Optional[list[dict]] = None,
    temperature: Optional[float] = None,
) -> ChatReply:
    """一次 chat 调用。失败重试（网络/限流），重试耗尽后向上抛。"""
    settings = get_settings()
    real_client = client if client is not None else get_client()
    model = settings.llm_model
    attempts = max(5, settings.llm_max_retries)  # 下限 5：配合指数退避跨过故障窗口
    last_err: Exception | None = None

    for attempt in range(attempts):
        if attempt:
            # 指数退避 5s/10s/20s/40s（封顶 60s）：等出 API 的慢查询窗口，
            # 紧挨着重试只会把三次机会全砸在同一个故障窗口里
            time.sleep(min(60.0, 5.0 * 2 ** (attempt - 1)))
        try:
            kwargs: dict = {
                "model": model,
                "messages": messages,
                "temperature": settings.llm_temperature if temperature is None else temperature,
            }
            if tools:
                kwargs["tools"] = tools
            resp = real_client.chat.completions.create(**kwargs)
            message = resp.choices[0].message
            usage = getattr(resp, "usage", None)
            return ChatReply(
                text=message.content or "",
                tool_calls=_normalize_tool_calls(getattr(message, "tool_calls", None)),
                prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            )
        except Exception as err:  # noqa: BLE001 - 网络/限流统一走重试
            last_err = err
            continue

    raise RuntimeError(f"chat 在 {attempts} 次尝试后仍失败：{last_err!r}")
