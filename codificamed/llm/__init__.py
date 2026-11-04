"""Fábrica de clientes LLM: el proveedor se elige por configuración (LLM_PROVIDER)."""
from .base import ClienteLLM, ErrorLLM, RespuestaLLM
from .proveedores import ClienteAnthropic, ClienteOllama, ClienteOpenAI

_CLIENTES = {"ollama": ClienteOllama, "anthropic": ClienteAnthropic, "openai": ClienteOpenAI}


def crear_cliente(settings) -> ClienteLLM:
    try:
        return _CLIENTES[settings.llm_provider](settings)
    except KeyError:
        raise ErrorLLM(f"Proveedor desconocido: {settings.llm_provider}. Use: {', '.join(_CLIENTES)}")


__all__ = ["ClienteLLM", "ErrorLLM", "RespuestaLLM", "crear_cliente"]
