from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Налаштування з оточення (INDEX_PATH=...) і з файлу .env. У коді нічого не зашито.

    Один і той самий образ Docker працює і локально, і на платформі: різниця лише в змінних.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    index_path: Path = Path("data/index.bin")
    # якщо корпус лежить не там, де був під час `findex index` (наприклад, у контейнері)
    corpus_root: Path | None = None
    cache_size: int = 256  # запитів у кеші Searcher
    text_cache: int = 128  # документів, чий текст тримаємо в пам'яті для сніпетів
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()
