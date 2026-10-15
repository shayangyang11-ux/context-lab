"""配置层：与 shop-agent 同一套纪律（12-Factor，前缀隔离）。"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="CONTEXT_LAB_",
        extra="ignore",
    )

    llm_base_url: str = "https://open.bigmodel.cn/api/paas/v4/"
    llm_api_key: str = ""
    # 实验默认 flash 档：25 轮 × 5 策略，成本敏感
    llm_model: str = "glm-4.5-flash"
    llm_temperature: float = 0.3
    # 2026-10-13 教训：flash 偶发数分钟级慢查询，60s×3 次快速重试全部
    # 落在同一个抖动窗口里，budget 策略整场报废。放宽为长超时 + 多次退避。
    llm_timeout: float = 180.0
    llm_min_request_interval: float = 1.0
    llm_max_retries: int = 5


@lru_cache
def get_settings() -> Settings:
    return Settings()
