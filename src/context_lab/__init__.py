"""context-lab：上下文管理三层策略 + 预算分配器，与 25 轮长任务 token 对比实验。

用法：
    python -m context_lab --dry-run     # 零费用验证管线
    python -m context_lab               # 真实 LLM 实验（读 .env 配置）
    python -m context_lab --report      # 从 results.json 重新生成报告与图表
"""

__version__ = "0.1.0"
