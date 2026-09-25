"""
Central configuration. Everything tunable lives here, sourced from .env,
so you never hardcode a model name or path deep in the code.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_provider: str = "anthropic"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-6"

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"

    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    data_dir: str = "data/sample"
    index_dir: str = "data/index"

    top_k_retrieve: int = 20
    top_k_final: int = 4
    hybrid_alpha: float = 0.5  # 0 = pure BM25, 1 = pure vector

    class Config:
        env_file = ".env"


settings = Settings()
