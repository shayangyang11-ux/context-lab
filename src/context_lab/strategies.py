"""上下文管理策略：三层机制 + 预算分配器（本仓库的核心资产）。
分层设计（系统提示与工具 schema 不归策略管 —— 它们是"第 0 层"，
由 agent 每轮无条件前置，这正是"不可压缩"原则的实现位置）：
  第 0 层  system prompt + 安全条款 + 工具 schema   —— 永不压缩
  第 1 层  Truncation  截断：保头保尾，硬切中段
  第 2 层  SlidingWindow 滑窗：只留最近 N 条
  第 3 层  Summary  摘要压缩：老对话压成要点，保留语义
  精化层  Budget  预算分配：每轮调用前按优先级装填 token 预算
           （摘要 > 近期对话 > 更老对话），pinned 决策点不可压缩
一个诚实的对比设计：压缩不是免费的 —— Summary / Budget 的压缩
调用本身要花 token（overhead），实验里单独记账，不藏账。
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Callable

from context_lab.messages import Msg, estimate_messages_cost

# 压缩器合同：被逐出的消息 + 上次摘要 -> (新摘要, 这次压缩花费的 token)
Compressor = Callable[[list[Msg], str], tuple[str, int]]

_MARKER = "（……中间 {n} 条历史已按策略截断……）"

# ==================== 【修改1：新增工具结果截断常量与函数】 ====================
TOOL_RESULT_LIMIT = 2000


def truncate_tool_output(text: str, keep: int = 900) -> str:
    """第 1 层：工具结果字符级截断 —— >2000 保头尾，中段显式标记。"""
    if len(text) <= TOOL_RESULT_LIMIT:
        return text
    omitted = len(text) - 2 * keep
    return (text[:keep]
            + f"\n（……中间 {omitted} 字符已截断，原文共 {len(text)} 字符……）\n"
            + text[-keep:])


# ==============================================================================

class ContextStrategy(ABC):
    """策略只管理历史消息；system / 工具 schema 由 agent 前置。"""
    name: str = "strategy"
    # ==================== 【修改2：基类新增工具截断开关属性】 ====================
    tool_result_limit: int | None = None

    @abstractmethod
    def add(self, msgs: list[Msg]) -> None: ...

    @abstractmethod
    def build(self) -> list[Msg]: ...

    def overhead_tokens(self) -> int:
        """管理开销：压缩调用累计花费（发送给模型的 prompt+completion）。"""
        return 0

    def reset(self) -> None:
        pass


# ------------------------------------------------------------- 第 1 层 --
class FullHistory(ContextStrategy):
    """基线：全量保留。长任务下每轮 prompt 线性增长，O(n²) 累计成本。"""
    name = "full"

    def __init__(self) -> None:
        self.history: list[Msg] = []

    def add(self, msgs: list[Msg]) -> None:
        self.history.extend(msgs)

    def build(self) -> list[Msg]:
        return list(self.history)

    def reset(self) -> None:
        self.history = []


class Truncation(ContextStrategy):
    """截断：保头（开场约束/早期事实）+ 保尾（近期上下文），硬切中段。"""
    name = "truncation"

    def __init__(self, keep_first: int = 4, keep_last: int = 10) -> None:
        self.keep_first = keep_first
        self.keep_last = keep_last
        self.history: list[Msg] = []

    def add(self, msgs: list[Msg]) -> None:
        self.history.extend(msgs)

    def build(self) -> list[Msg]:
        n = len(self.history)
        if n <= self.keep_first + self.keep_last:
            return list(self.history)
        dropped = n - self.keep_first - self.keep_last
        marker = Msg("system", _MARKER.format(n=dropped))
        return self.history[: self.keep_first] + [marker] + self.history[-self.keep_last:]

    def reset(self) -> None:
        self.history = []


class SlidingWindow(ContextStrategy):
    """滑窗：只留最近 N 条。最便宜，但早期事实直接丢失——pinned 也保不住，
    这是"朴素滑窗违反不可压缩原则"的对照组，实验的意义就在这组数字里。"""
    name = "sliding"

    def __init__(self, max_messages: int = 10) -> None:
        self.max_messages = max_messages
        self.history: list[Msg] = []

    def add(self, msgs: list[Msg]) -> None:
        self.history.extend(msgs)

    def build(self) -> list[Msg]:
        window = self.history[-self.max_messages:]
        dropped = len(self.history) - len(window)
        if not dropped:
            return list(window)
        marker = Msg("system", _MARKER.format(n=dropped))
        return [marker] + window

    def reset(self) -> None:
        self.history = []


# ------------------------------------------------------------- 第 3 层 --
class Summary(ContextStrategy):
    """摘要压缩：窗口外的老消息合并进滚动摘要（语义保留，花压缩成本）。"""
    name = "summary"
    # ==================== 【修改3：启用工具结果截断】 ====================
    tool_result_limit = 2000

    def __init__(self, window: int, compressor: Compressor) -> None:
        self.window = window
        self.compressor = compressor
        self.history: list[Msg] = []
        self.summary: str = ""
        self._overhead = 0

    def add(self, msgs: list[Msg]) -> None:
        self.history.extend(msgs)
        self._evict()

    def _evict(self) -> None:
        if len(self.history) <= self.window:
            return
        evicted = self.history[: -self.window]
        self.history = self.history[-self.window:]
        self.summary, cost = self.compressor(evicted, self.summary)
        self._overhead += cost

    def build(self) -> list[Msg]:
        parts: list[Msg] = []
        if self.summary:
            parts.append(Msg("system", f"前情摘要（早期对话压缩）：{self.summary}"))
        return parts + list(self.history)

    def overhead_tokens(self) -> int:
        return self._overhead

    def reset(self) -> None:
        self.history = []
        self.summary = ""
        self._overhead = 0


# ------------------------------------------------------------- 精化层 --
class Budget(ContextStrategy):
    """预算分配：每轮 build() 前按优先级装填固定 token 预算。
    优先级（这就是"每轮调用前分配 token 预算"的落地）：
      1. 滚动摘要 —— 早期事实的语义载体，最高优先
      2. 近期对话 —— 从新到旧装填，直到预算耗尽
      3. 装不下的更老对话 —— 压进摘要（花压缩成本）
    例外条款（不可压缩原则）：
      - pinned 消息（关键决策点：改地址确认、预算变更等）恒保留，
        允许小幅超预算 —— 代价显式记录在 pinned_overhead 字段里
      - system / 工具 schema 在第 0 层，不进本预算（agent 前置）
    """
    name = "budget"
    # ==================== 【修改4：启用工具结果截断】 ====================
    tool_result_limit = 2000

    def __init__(self, budget_tokens: int, compressor: Compressor) -> None:
        self.budget = budget_tokens
        self.compressor = compressor
        self.history: list[Msg] = []
        self.summary: str = ""
        self._overhead = 0
        self._last_evicted_key: tuple | None = None
        self.pinned_overhead = 0  # pinned 顶破预算的量（观测用）

    def add(self, msgs: list[Msg]) -> None:
        self.history.extend(msgs)

    def _summary_cost(self) -> int:
        if not self.summary:
            return 0
        from context_lab.messages import estimate_tokens, PER_MESSAGE_OVERHEAD
        return estimate_tokens(f"前情摘要：{self.summary}") + PER_MESSAGE_OVERHEAD

    def _select(self) -> tuple[list[int], list[int]]:
        """返回 (保留索引, 逐出索引)。从新到旧装填，pinned 恒留。"""
        kept: set[int] = set()
        used = self._summary_cost()
        for idx in range(len(self.history) - 1, -1, -1):
            m = self.history[idx]
            cost = estimate_messages_cost([m])
            if m.pinned:
                kept.add(idx)
                used += cost
                continue
            if used + cost <= self.budget:
                kept.add(idx)
                used += cost

        kept_list = sorted(kept)
        evicted_list = [i for i in range(len(self.history)) if i not in kept]
        return kept_list, evicted_list

    def build(self) -> list[Msg]:
        kept_list, evicted_list = self._select()
        key = tuple(evicted_list)
        if evicted_list and key != self._last_evicted_key:
            evicted = [self.history[i] for i in evicted_list]
            self.summary, cost = self.compressor(evicted, self.summary)
            self._overhead += cost
            self._last_evicted_key = key
            # 摘要变了，保留集要按新摘要成本重算一次
            kept_list, evicted_list = self._select()

        self.pinned_overhead = max(
            0, estimate_messages_cost([self.history[i] for i in kept_list])
            + self._summary_cost()
            - self.budget
        )

        parts: list[Msg] = []
        if self.summary:
            parts.append(Msg("system", f"前情摘要（早期对话压缩）：{self.summary}"))
        return parts + [self.history[i] for i in kept_list]

    def overhead_tokens(self) -> int:
        return self._overhead

    def reset(self) -> None:
        self.history = []
        self.summary = ""
        self._overhead = 0
        self._last_evicted_key = None
        self.pinned_overhead = 0
