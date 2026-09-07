# backend configuration
import os
from functools import lru_cache
from typing import Optional
from pydantic import Field, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    # db settings
    postgres_host: str = Field(default="postgres", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5432, alias="POSTGRES_PORT")
    postgres_user: str = Field(default="emqx", alias="POSTGRES_USER")
    postgres_password: str = Field(default="emqx", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="emqx", alias="POSTGRES_DB")
    database_url: Optional[str] = Field(default=None, alias="DATABASE_URL")

    # mqtt connection
    mqtt_broker: str = Field(default="emqx", alias="MQTT_BROKER")
    mqtt_port: int = Field(default=1883, alias="MQTT_PORT")
    mqtt_topic_prefix: str = Field(default="v1", alias="MQTT_TOPIC_PREFIX")

    # clearml tracking
    clearml_api_host: str = Field(default="http://localhost:30908", alias="CLEARML_API_HOST")
    clearml_web_host: str = Field(default="http://localhost:30988", alias="CLEARML_WEB_HOST")
    clearml_files_host: str = Field(default="http://localhost:30981", alias="CLEARML_FILES_HOST")
    clearml_project_name: str = Field(default="research-ml-edge", alias="CLEARML_PROJECT_NAME")
    clearml_access_key: str = Field(default="", alias="CLEARML_API_ACCESS_KEY")
    clearml_secret_key: str = Field(default="", alias="CLEARML_API_SECRET_KEY")

    # nas and penalty params
    nas_data_threshold_bytes: int = Field(default=1048576, alias="NAS_DATA_THRESHOLD_BYTES")
    ram_limit_bytes: int = Field(default=4194304, alias="RAM_LIMIT_BYTES")
    sd_latency_penalty_factor: float = Field(default=0.05, alias="SD_LATENCY_PENALTY_FACTOR")

    # api host and port
    api_host: str = Field(default="0.0.0.0", alias="API_HOST")
    api_port: int = Field(default=8000, alias="API_PORT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    @computed_field
    @property
    def dsn(self) -> str:
        # build asyncpg dsn
        if self.database_url:
            return self.database_url
        return f"postgresql://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

@lru_cache()
def get_settings() -> Settings:
    # return cached settings instance
    return Settings()
