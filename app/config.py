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
    llm_provider: str = "gemini"  # "gemini" | "ollama" | "groq"
    ollama_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # Groq (free tier) — set LLM_PROVIDER=groq to use
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"

    # Vector DB (Qdrant Cloud)
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "flux_documents"

    # Reranker (CrossEncoder)
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"
    enable_reranking: bool = True

    # Web Search
    tavily_api_key: str = ""
    openalex_base_url: str = "https://api.openalex.org"

    # Router LLM (can use a different/faster model for classification)
    router_llm_model: str = ""  # empty = same as llm_model

    # Agent LLM (needs tool-calling support — NOT flash-lite)
    agent_llm_model: str = "gemini-2.0-flash"
    agent_llm_provider: str = ""  # empty = same as llm_provider

    # Internal API (FLux-AI → RPM-BE for agent tool calls)
    rpm_be_url: str = "http://localhost:3000"
    internal_api_key: str = ""

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


def get_llm(temperature: float = 0.7, model: str | None = None, provider: str | None = None):
    """
    LLM factory — returns the active chat model.

    Configure via .env:
        LLM_PROVIDER=gemini   # default — uses Google Gemini
        LLM_PROVIDER=ollama   # local Ollama server
        LLM_PROVIDER=groq     # Groq cloud (free tier)
        LLM_MODEL=llama3.2    # any model pulled with `ollama pull <model>`
        OLLAMA_BASE_URL=http://localhost:11434
    """
    settings = get_settings()
    active_provider = provider or settings.llm_provider

    if active_provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model or settings.ollama_model,
            base_url=settings.ollama_base_url,
            temperature=temperature,
        )

    if active_provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=model or settings.groq_model,
            api_key=settings.groq_api_key,
            temperature=temperature,
        )

    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(
        model=model or settings.llm_model,
        google_api_key=settings.gemini_api_key,
        temperature=temperature,
    )


def get_router_llm():
    """
    LLM factory for the Supervisor router — deterministic (temp=0).

    Uses ROUTER_LLM_MODEL if set, otherwise falls back to the main LLM_MODEL.
    This lets you use a faster/cheaper model for routing decisions.
    """
    settings = get_settings()
    model = settings.router_llm_model or None  # empty string → None → use default
    return get_llm(temperature=0, model=model)


def get_agent_llm():
    """
    LLM factory for the Action Agent — needs tool-calling support.

    Uses AGENT_LLM_MODEL (default: gemini-2.0-flash) and optionally
    AGENT_LLM_PROVIDER if you want the agent to use a different provider
    (e.g., groq for free tier).
    """
    settings = get_settings()
    provider = settings.agent_llm_provider or None
    model = settings.agent_llm_model or None
    return get_llm(temperature=0.3, model=model, provider=provider)
