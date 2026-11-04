"""Configuración central. Todo se lee desde variables de entorno / archivo .env.

Cambiar de modelo local a API externa es solo cambiar LLM_PROVIDER en .env
(o pasar --provider en los scripts / elegirlo en la interfaz).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
CACHE_DIR = ROOT / ".cache"

load_dotenv(ROOT / ".env")

PROVIDERS = ("ollama", "anthropic", "openai", "rag")


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


@dataclass(frozen=True)
class Settings:
    # Proveedor de LLM: ollama (local) | anthropic | openai | rag (línea base sin LLM)
    llm_provider: str = field(default_factory=lambda: _env("LLM_PROVIDER", "ollama"))

    # Ollama (modelo local)
    ollama_host: str = field(default_factory=lambda: _env("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: _env("OLLAMA_MODEL", "qwen2.5:7b"))

    # Anthropic (Claude)
    anthropic_api_key: str = field(default_factory=lambda: _env("ANTHROPIC_API_KEY"))
    anthropic_model: str = field(default_factory=lambda: _env("ANTHROPIC_MODEL", "claude-haiku-5-5"))

    # OpenAI u otra API compatible con OpenAI (Groq, OpenRouter, etc.)
    openai_api_key: str = field(default_factory=lambda: _env("OPENAI_API_KEY"))
    openai_model: str = field(default_factory=lambda: _env("OPENAI_MODEL", "gpt-4o-mini"))
    openai_base_url: str = field(default_factory=lambda: _env("OPENAI_BASE_URL"))

    # Generación
    temperature: float = field(default_factory=lambda: float(_env("LLM_TEMPERATURE", "0")))
    max_tokens: int = field(default_factory=lambda: int(_env("LLM_MAX_TOKENS", "2048")))
    timeout_s: int = field(default_factory=lambda: int(_env("LLM_TIMEOUT", "300")))

    # Recuperación (RAG)
    embedding_backend: str = field(default_factory=lambda: _env("EMBEDDING_BACKEND", "hibrido"))
    embedding_model: str = field(
        default_factory=lambda: _env("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    )
    top_k_por_mencion: int = field(default_factory=lambda: int(_env("TOP_K", "8")))
    max_candidatos: int = field(default_factory=lambda: int(_env("MAX_CANDIDATOS", "60")))
    # Umbral de similitud para la línea base sin LLM (vacío = automático según el backend)
    umbral_baseline: float | None = field(
        default_factory=lambda: float(_env("UMBRAL_BASELINE")) if _env("UMBRAL_BASELINE") else None
    )

    # Costos opcionales (USD por millón de tokens) para estimar el gasto de la API
    precio_entrada_mtok: float = field(default_factory=lambda: float(_env("PRECIO_ENTRADA_MTOK", "0")))
    precio_salida_mtok: float = field(default_factory=lambda: float(_env("PRECIO_SALIDA_MTOK", "0")))

    def con(self, **cambios) -> "Settings":
        """Devuelve una copia con algunos valores cambiados (p. ej. otro proveedor)."""
        return replace(self, **cambios)

    @property
    def modelo_activo(self) -> str:
        return {
            "ollama": self.ollama_model,
            "anthropic": self.anthropic_model,
            "openai": self.openai_model,
            "rag": f"rag-{self.embedding_backend}",
        }.get(self.llm_provider, "?")


def get_settings(**cambios) -> Settings:
    s = Settings()
    return s.con(**cambios) if cambios else s
