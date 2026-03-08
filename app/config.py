"""Centralized configuration using Pydantic Settings."""

from functools import lru_cache

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings, loaded from environment variables / .env file."""

    # LLM
    gemini_api_key: str = ""
    llm_model: str = "gemini-2.0-flash-lite"
    embedding_model: str = "text-embedding-004"

    # Local LLM via Ollama — set LLM_PROVIDER=ollama to use
    llm_provider: str = "gemini"  # "gemini" | "ollama"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # Vector DB (Qdrant Cloud)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "flux_documents"
    # Redis
    redis_url: str = "redis://localhost:6379"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # CORS
    cors_origins: str = "http://localhost:5173,http://localhost:2916,http://localhost:3000"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()


def get_llm(temperature: float = 0.7):
    """
    LLM factory — returns the active chat model.

    Configure via .env:
        LLM_PROVIDER=gemini   # default — uses Google Gemini
        LLM_PROVIDER=ollama   # local Ollama server
        LLM_MODEL=llama3.2    # any model pulled with `ollama pull <model>`
        OLLAMA_BASE_URL=http://localhost:11434
    """
    settings = get_settings()
    if settings.llm_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=settings.llm_model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
    )
