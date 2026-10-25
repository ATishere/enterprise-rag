from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central config — reads from .env + env vars, validates at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "enterprise-rag"
    environment: str = "development"
    debug: bool = False

    postgres_url: str = Field(
        default="postgresql+asyncpg://user:pass@localhost:5432/rag"
    )

    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    embedding_model: str = "BAAI/bge-large-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-large"
    top_k_retrieval: int = 20
    top_k_rerank: int = 5

    anthropic_api_key: str | None = None
    llm_model: str = "claude-3-5-sonnet-20241022"

    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @model_validator(mode="after")
    def _check_production_requirements(self) -> "Settings":
        """Require external service keys before starting in production."""
        if self.environment == "production":
            missing = []
            if not self.anthropic_api_key:
                missing.append("ANTHROPIC_API_KEY")
            if not self.langfuse_public_key:
                missing.append("LANGFUSE_PUBLIC_KEY")
            if not self.langfuse_secret_key:
                missing.append("LANGFUSE_SECRET_KEY")
            if missing:
                raise ValueError(
                    f"Missing required env vars in production: {', '.join(missing)}"
                )
        return self


settings = Settings()
