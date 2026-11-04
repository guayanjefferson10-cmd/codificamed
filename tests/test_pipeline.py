"""Prueba el pipeline completo con un servidor Ollama falso (no requiere modelos)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from codificamed.config import get_settings
from codificamed.pipeline import CodificaMed, localizar_evidencia

RESPUESTA = {"codigos": [
    {"codigo": "I10", "evidencia": "HTA en tto", "estado": "confirmado", "justificacion": "HTA"},
    {"codigo": "e119", "evidencia": "DM2", "estado": "confirmado"},
    # error deliberado del LLM: confirma algo negado -> NegEx debe corregirlo
    {"codigo": "R50.9", "evidencia": "Niega fiebre", "estado": "confirmado"},
    {"codigo": "U99.9", "evidencia": "inventado", "estado": "confirmado"},
]}


class FakeOllama(BaseHTTPRequestHandler):
    ultimo = {}

    def do_POST(self):
        cuerpo = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.ultimo = cuerpo
        out = json.dumps({"message": {"content": json.dumps(RESPUESTA)},
                          "prompt_eval_count": 900, "eval_count": 120}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def servidor():
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_pipeline_ollama(servidor):
    s = get_settings(llm_provider="ollama", ollama_host=servidor, embedding_backend="tfidf")
    r = CodificaMed(s, negacion="hibrida").codificar("Pcte con HTA en tto y DM2. Niega fiebre.")
    por_cod = {c.codigo: c for c in r.codigos}
    assert por_cod["I10"].estado == "confirmado" and por_cod["I10"].inicio is not None
    assert "E11.9" in por_cod                          # normalizado desde 'e119'
    assert por_cod["R50.9"].estado == "negado"         # corregido por NegEx
    assert por_cod["R50.9"].ajustado_por_negex
    assert "U99.9" not in por_cod and any("U99.9" in a for a in r.advertencias)
    assert r.tokens_entrada == 900
    prompt = FakeOllama.ultimo["messages"][1]["content"]
    assert "[NEG fiebre]" in prompt and "I10" in prompt
    assert FakeOllama.ultimo["options"]["num_ctx"] >= 8192


def test_politica_solo_llm(servidor):
    s = get_settings(llm_provider="ollama", ollama_host=servidor, embedding_backend="tfidf")
    r = CodificaMed(s, negacion="llm").codificar("Pcte con HTA en tto y DM2. Niega fiebre.")
    assert {c.codigo: c for c in r.codigos}["R50.9"].estado == "confirmado"


def test_linea_base():
    s = get_settings(llm_provider="rag", embedding_backend="tfidf")
    r = CodificaMed(s).codificar("Dx: apendicitis aguda. Se descarta embarazo ectópico.")
    assert "K35.9" in {c.codigo for c in r.confirmados} or "K35" in {c.codigo for c in r.confirmados}


def test_localizar_evidencia_aproximada():
    nota = "Paciente con hipertension arterial en tratamiento"
    assert localizar_evidencia(nota, "hipertensión arterial") is not None


def test_negados_negex_y_evidencia_aproximada(servidor):
    s = get_settings(llm_provider="ollama", ollama_host=servidor, embedding_backend="tfidf")
    r = CodificaMed(s).codificar("Pcte con HTA y DM2 en tto. Niega fiebre. Se descarta neumonía.")
    i10 = {c.codigo: c for c in r.codigos}["I10"]
    assert i10.inicio is not None            # "HTA en tto" ubicado aunque no es literal
    negados = [c for c in r.codigos if c.estado == "negado"]
    assert any("neumonía" in c.evidencia for c in negados)   # agregado por NegEx
    assert all(c.estado != "confirmado" for c in negados)
