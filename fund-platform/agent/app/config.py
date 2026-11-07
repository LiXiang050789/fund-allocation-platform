"""环境变量与路径配置。所有密钥只从 .env 读取，不写死、不入库。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

AGENT_DIR = Path(__file__).resolve().parents[1]
PLATFORM_DIR = AGENT_DIR.parent

load_dotenv(AGENT_DIR / ".env")


def _flag(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # LLM
    deepseek_api_key: str = field(default_factory=lambda: os.getenv("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(
        default_factory=lambda: os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    )
    deepseek_model: str = field(default_factory=lambda: os.getenv("DEEPSEEK_MODEL", "deepseek-flash"))
    llm_temperature: float = field(default_factory=lambda: float(os.getenv("LLM_TEMPERATURE", "0.2")))
    llm_timeout: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT_SECONDS", "60")))

    # 飞书
    feishu_app_id: str = field(default_factory=lambda: os.getenv("FEISHU_APP_ID", ""))
    feishu_app_secret: str = field(default_factory=lambda: os.getenv("FEISHU_APP_SECRET", ""))
    feishu_chat_id: str = field(default_factory=lambda: os.getenv("FEISHU_CHAT_ID", ""))
    feishu_webhook_url: str = field(default_factory=lambda: os.getenv("FEISHU_WEBHOOK_URL", ""))
    feishu_webhook_secret: str = field(default_factory=lambda: os.getenv("FEISHU_WEBHOOK_SECRET", ""))

    # 开关
    enable_vector_rag: bool = field(default_factory=lambda: _flag("ENABLE_VECTOR_RAG", "1"))
    enable_llm_verifier: bool = field(default_factory=lambda: _flag("ENABLE_LLM_VERIFIER", "1"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    # 路径
    agent_dir: Path = AGENT_DIR
    platform_dir: Path = PLATFORM_DIR
    score_path: Path = AGENT_DIR.parent / "data/clean/market_score_daily.csv"
    index_dir: Path = AGENT_DIR / "data/index"
    reports_dir: Path = AGENT_DIR / "reports"

    @property
    def feishu_app_enabled(self) -> bool:
        return bool(self.feishu_app_id and self.feishu_app_secret)

    @property
    def feishu_webhook_enabled(self) -> bool:
        return bool(self.feishu_webhook_url)


settings = Settings()
