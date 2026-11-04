"""Interfaz común para todos los proveedores de LLM."""
from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field


class ErrorLLM(RuntimeError):
    pass


@dataclass
class RespuestaLLM:
    proveedor: str
    modelo: str
    texto: str
    datos: dict
    latencia_s: float
    tokens_entrada: int = 0
    tokens_salida: int = 0
    intentos: int = 1
    extra: dict = field(default_factory=dict)


def extraer_json(texto: str) -> dict:
    """Extrae el primer objeto JSON del texto (tolera ```json ... ``` y texto alrededor)."""
    t = texto.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    ini, fin = t.find("{"), t.rfind("}")
    if ini >= 0 and fin > ini:
        return json.loads(t[ini:fin + 1])
    raise ValueError("La respuesta no contiene JSON")


class ClienteLLM(ABC):
    proveedor = "base"

    def __init__(self, settings):
        self.s = settings

    @property
    @abstractmethod
    def modelo(self) -> str: ...

    @abstractmethod
    def _llamar(self, sistema: str, usuario: str, esquema: dict) -> tuple[str, int, int]:
        """Devuelve (texto, tokens_entrada, tokens_salida)."""

    def generar_json(self, sistema: str, usuario: str, esquema: dict, reintentos: int = 1) -> RespuestaLLM:
        t0 = time.perf_counter()
        tin = tout = 0
        ultimo_error = None
        for intento in range(1, reintentos + 2):
            texto, a, b = self._llamar(sistema, usuario, esquema)
            tin, tout = tin + a, tout + b
            try:
                datos = extraer_json(texto)
                return RespuestaLLM(self.proveedor, self.modelo, texto, datos,
                                    time.perf_counter() - t0, tin, tout, intento)
            except (ValueError, json.JSONDecodeError) as e:
                ultimo_error = e
                usuario = usuario + "\n\nIMPORTANTE: responde SOLO con un objeto JSON válido."
        raise ErrorLLM(f"{self.proveedor}: respuesta no es JSON válido ({ultimo_error})")
