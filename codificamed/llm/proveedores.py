"""Implementaciones de los proveedores: Ollama (local), Anthropic (Claude) y OpenAI/compatibles."""
from __future__ import annotations

import requests

from .base import ClienteLLM, ErrorLLM


class ClienteOllama(ClienteLLM):
    """Modelo local servido por Ollama (https://ollama.com). No envía datos fuera del equipo."""

    proveedor = "ollama"

    @property
    def modelo(self) -> str:
        return self.s.ollama_model

    def _llamar(self, sistema, usuario, esquema):
        url = self.s.ollama_host.rstrip("/") + "/api/chat"
        cuerpo = {
            "model": self.modelo,
            "messages": [{"role": "system", "content": sistema},
                         {"role": "user", "content": usuario}],
            "stream": False,
            "format": esquema,  # salida estructurada (Ollama >= 0.5)
            "options": {"temperature": self.s.temperature, "num_ctx": 8192,
                        "num_predict": self.s.max_tokens},
        }
        try:
            r = requests.post(url, json=cuerpo, timeout=self.s.timeout_s)
        except requests.ConnectionError as e:
            raise ErrorLLM(f"No se pudo conectar con Ollama en {self.s.ollama_host}. "
                           "¿Está corriendo `ollama serve`?") from e
        if r.status_code == 404:
            raise ErrorLLM(f"Ollama no tiene el modelo '{self.modelo}'. "
                           f"Ejecuta: ollama pull {self.modelo}")
        if r.status_code >= 400 and isinstance(cuerpo["format"], dict):
            # versiones antiguas de Ollama no aceptan un esquema: se pide JSON genérico
            cuerpo["format"] = "json"
            r = requests.post(url, json=cuerpo, timeout=self.s.timeout_s)
        r.raise_for_status()
        d = r.json()
        return d["message"]["content"], d.get("prompt_eval_count", 0), d.get("eval_count", 0)


class ClienteAnthropic(ClienteLLM):
    """API de Claude (Anthropic)."""

    proveedor = "anthropic"

    def __init__(self, settings):
        super().__init__(settings)
        if not settings.anthropic_api_key:
            raise ErrorLLM("Falta ANTHROPIC_API_KEY en .env")
        import anthropic

        self._c = anthropic.Anthropic(api_key=settings.anthropic_api_key, timeout=settings.timeout_s, max_retries=8)

    @property
    def modelo(self) -> str:
        return self.s.anthropic_model

    def _llamar(self, sistema, usuario, esquema):
        import anthropic

        args = dict(model=self.modelo, max_tokens=self.s.max_tokens, system=sistema,
                    messages=[{"role": "user", "content": usuario}],
                    temperature=self.s.temperature)
        try:
            msg = self._c.messages.create(**args)
        except anthropic.BadRequestError as e:
            if "temperature" not in str(e):
                raise
            args.pop("temperature")  # algunos modelos no aceptan este parámetro
            msg = self._c.messages.create(**args)
        texto = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        return texto, msg.usage.input_tokens, msg.usage.output_tokens


class ClienteOpenAI(ClienteLLM):
    """API de OpenAI o cualquier API compatible (Groq, OpenRouter, DeepSeek...) vía OPENAI_BASE_URL."""

    proveedor = "openai"

    def __init__(self, settings):
        super().__init__(settings)
        if not settings.openai_api_key:
            raise ErrorLLM("Falta OPENAI_API_KEY en .env")
        from openai import OpenAI

        self._c = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url or None,
                         timeout=settings.timeout_s, max_retries=8)

    @property
    def modelo(self) -> str:
        return self.s.openai_model

    def _llamar(self, sistema, usuario, esquema):
        from openai import BadRequestError

        args = dict(model=self.modelo, temperature=self.s.temperature, max_tokens=self.s.max_tokens,
                    response_format={"type": "json_object"},
                    messages=[{"role": "system", "content": sistema},
                              {"role": "user", "content": usuario}])
        try:
            r = self._c.chat.completions.create(**args)
        except BadRequestError as e:
            if "response_format" not in str(e) and "json" not in str(e).lower():
                raise
            # algunas API compatibles no aceptan json_object: se pide JSON solo por el prompt
            args.pop("response_format")
            r = self._c.chat.completions.create(**args)
        u = r.usage
        return r.choices[0].message.content or "", getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)
