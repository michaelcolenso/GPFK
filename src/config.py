from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "sqlite:///./harbinger.db"

    opensky_username: str = ""
    opensky_password: str = ""

    edgar_user_agent: str = "Harbinger Research contact@harbinger.local"

    api_secret_key: str = "dev-secret-key"
    rate_limit_per_minute: int = 60

    edgar_collect_cron: str = "0 */4 * * *"
    flight_collect_cron: str = "*/30 * * * *"
    jobs_collect_cron: str = "0 8 * * *"

    # Lead time window: how many days before announcement we look for signals
    signal_lookback_days: int = 90

    # Model cache path
    model_cache_dir: str = ".model_cache"


@lru_cache
def get_settings() -> Settings:
    return Settings()
