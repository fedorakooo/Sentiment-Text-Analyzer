from pydantic_settings import BaseSettings, SettingsConfigDict


class OllamaSettings(BaseSettings):
    """Ollama connection settings."""

    base_url: str = "http://ollama:11434"
    model: str = "llama3"
    request_timeout: int = 30

    model_config = SettingsConfigDict(env_prefix="OLLAMA_", env_file=".env", extra="ignore")


class RedisSettings(BaseSettings):
    """Redis cache settings."""

    host: str = "redis"
    port: int = 6379
    db: int = 0
    user: str | None = None
    user_password: str | None = None
    redis_ttl: int = 3600

    @property
    def url(self) -> str:
        return f"redis://{self.user}:{self.user_password}@{self.host}:{self.port}/{self.db}"

    model_config = SettingsConfigDict(env_prefix="REDIS_", env_file=".env", extra="ignore")


class Settings(BaseSettings):
    """Application settings container."""

    ollama: OllamaSettings = OllamaSettings()
    redis: RedisSettings = RedisSettings()

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
