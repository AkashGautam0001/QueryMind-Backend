from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_File_encoding="utf-8",
        extra="ignore"
    )

    app_name: str = "RAG System"
    environment: str = "development"
    debug: bool = False
    
    postgres_user: str = "raguser"
    postgres_password: str = "ragpassword"
    postgres_db: str = "ragdb"
    postgres_host: str = "postgres"
    postgres_port: str = 5432

    qdrant_host: str = "qdrant"
    qdrant_port: str = 6333

    redis_host: str = "redis"
    redis_port: int = 6379

    minio_endpoint: str = "minio:9000"
    minio_access_key: str = "ragminio"
    minio_secret_key: str = "ragminiosecret"
    minio_bucket: str = "rag-documents"
    minio_secure: bool = False

    openai_api_key: str = ""
    anthropic_api_key: str = ""

    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    embedding_batch_size: int = 100

    default_chunking_strategy: str = "parent_child"
    max_upload_size_mb: int = 50

    retrieval_prefetch_limit: int = 50

    rerank_model_name: str = "BAAI/bge-reranker-v2-m3"
    rerank_candidate_pool_size: int = 30
    rerank_final_top_n: int = 6
    context_max_tokens: int = 3000

    # Below this rerank score, a chunk is treated as not actually relevant.
    # bge-reranker-v2-m3 scores aren't calibrated probabilities, so this is
    # a corpus/model-specific tuning knob, not a universal cutoff — verify
    # it empirically against your own data rather than trusting the default.
    relevance_score_threshold: float = 0.0

    generation_model: str = "claude-sonnet-4-6"
    generation_max_tokens: int = 1024
    generation_temperature: float = 0.0
    conversation_history_max_messages: int = 10

    api_key: str = ""  # If set, all requests must include X-API-Key header
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    rate_limit_per_minute: int = 60

    semantic_cache_ttl: int = 3600
    semantic_cache_similarity_threshold: float = 0.92

    otlp_endpoint: str = ""  # e.g. http://jaeger:4317

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]
    
    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def redis_url(self) -> str:
        return f"redis://{self.redis_host}:{self.redis_port}/0"
    
    @property
    def minio_endpoint_url(self) -> str:
        scheme = "https" if self.minio_secure else "http"
        return f"{scheme}://{self.minio_endpoint}"
    
    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"
    
@lru_cache
def get_settings() -> Settings:
    return Settings()
