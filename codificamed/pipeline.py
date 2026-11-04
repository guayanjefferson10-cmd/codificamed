"""Pipeline completo de CodificaMed.

    nota en texto libre
      -> preprocesamiento (limpieza, segmentación, abreviaturas, ortografía)
      -> NegEx-es (negación / sospecha por token)
      -> recuperación semántica de candidatos CIE-10 (RAG)
      -> LLM (Ollama local o API externa) elige códigos + evidencia + estado
      -> validación: código en catálogo, evidencia localizada en la nota, coherencia con NegEx
      -> lista de códigos sugeridos para que el codificador humano valide
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import asdict, dataclass, field

from rapidfuzz import fuzz

from .abreviaturas import ABREVIATURAS
from .catalog import Catalogo, cargar_catalogo, normalizar_codigo
from .config import Settings, get_settings
from .llm import ErrorLLM, crear_cliente
from .negacion import AFIRMADO, NEGADO, SOSPECHA
from .preprocesamiento import NotaProcesada, a_consulta, preprocesar
from .prompts import ESQUEMA, SISTEMA, construir_usuario
from .recuperador import Candidato, Recuperador

CONFIRMADO = "confirmado"
_A_ESTADO = {AFIRMADO: CONFIRMADO, NEGADO: "negado", SOSPECHA: "sospecha"}


@dataclass
class CodigoSugerido:
    codigo: str
    descripcion: str
    estado: str                      # confirmado | negado | sospecha
    evidencia: str
    inicio: int | None = None        # offsets de la evidencia en la nota original
    fin: int | None = None
    justificacion: str = ""
    en_candidatos: bool = True
    estado_negex: str | None = None  # estado según NegEx sobre la evidencia
    estado_llm: str | None = None    # estado que propuso el LLM
    ajustado_por_negex: bool = False
    codigo_original: str = ""


@dataclass
class Resultado:
    proveedor: str
    modelo: str
    codigos: list[CodigoSugerido]
    candidatos: list[Candidato]
    latencia_total_s: float
    latencia_llm_s: float = 0.0
    tokens_entrada: int = 0
    tokens_salida: int = 0
    advertencias: list[str] = field(default_factory=list)
    nota: NotaProcesada | None = None
    respuesta_cruda: str = ""

    @property
    def confirmados(self) -> list[CodigoSugerido]:
        return [c for c in self.codigos if c.estado == CONFIRMADO]

    def a_dict(self) -> dict:
        return {
            "proveedor": self.proveedor, "modelo": self.modelo,
            "codigos": [asdict(c) for c in self.codigos],
            "candidatos": [asdict(c) for c in self.candidatos],
            "latencia_total_s": round(self.latencia_total_s, 3),
            "latencia_llm_s": round(self.latencia_llm_s, 3),
            "tokens_entrada": self.tokens_entrada, "tokens_salida": self.tokens_salida,
            "advertencias": self.advertencias,
        }


# ------------------------------------------------------------------ utilidades
def localizar_evidencia(texto: str, evidencia: str, umbral: float = 80) -> tuple[int, int] | None:
    """Ubica la evidencia en la nota (exacta, sin mayúsculas, o aproximada)."""
    ev = (evidencia or "").strip().strip('"\'«»“”.')
    if not ev:
        return None
    i = texto.find(ev)
    if i < 0:
        i = texto.lower().find(ev.lower())
    if i >= 0:
        return i, i + len(ev)
    al = fuzz.partial_ratio_alignment(ev.lower(), texto.lower())
    if al and al.score >= umbral:
        return al.dest_start, al.dest_end
    # Último recurso: la palabra más larga de la evidencia que sí aparece en la nota
    # (p. ej. el modelo escribe "HTA en tto" y la nota dice "HTA y DM2 en tto").
    palabras = sorted({w for w in re.findall(r"\w+", ev) if len(w) >= 3}, key=len, reverse=True)
    for w in palabras:
        m = re.search(rf"(?<!\w){re.escape(w)}(?!\w)", texto, re.IGNORECASE)
        if m:
            return m.start(), m.end()
    return None


def abreviaturas_en(texto: str) -> dict[str, str]:
    encontradas = {}
    for sigla, exp in ABREVIATURAS.items():
        if re.search(rf"(?<![\w]){re.escape(sigla)}(?![\w])", texto, re.IGNORECASE):
            encontradas[sigla] = exp
    return encontradas


# ------------------------------------------------------------------ pipeline
class CodificaMed:
    def __init__(self, settings: Settings | None = None, recuperador: Recuperador | None = None,
                 catalogo: Catalogo | None = None, negacion: str | None = None):
        self.s = settings or get_settings()
        self.catalogo = catalogo or cargar_catalogo()
        self.recuperador = recuperador or Recuperador(self.s, self.catalogo)
        # Política de negación: "hibrida" (NegEx corrige al LLM) o "llm" (solo el LLM decide)
        self.negacion = negacion or os.getenv("NEGACION_POLITICA", "hibrida")
        self._cliente = None

    @property
    def proveedor(self) -> str:
        return self.s.llm_provider

    def cliente(self):
        if self._cliente is None:
            self._cliente = crear_cliente(self.s)
        return self._cliente

    # -------------------------------------------------- recuperación de candidatos
    def candidatos(self, nota: NotaProcesada) -> list[Candidato]:
        clausulas = [c for c in nota.clausulas if len(re.sub(r"[^a-záéíóúñ]", "", c.consulta.lower())) >= 3]
        resultados = self.recuperador.buscar([c.consulta for c in clausulas], self.s.top_k_por_mencion)
        mejores: dict[str, Candidato] = {}
        garantizados: list[str] = []
        for res in resultados:
            for j, cand in enumerate(res):
                if cand.codigo not in mejores or cand.score > mejores[cand.codigo].score:
                    mejores[cand.codigo] = cand
                if j < 3:
                    garantizados.append(cand.codigo)
        orden = sorted(mejores.values(), key=lambda c: -c.score)
        elegidos = list(dict.fromkeys(garantizados))[: self.s.max_candidatos]
        for c in orden:
            if len(elegidos) >= self.s.max_candidatos:
                break
            if c.codigo not in elegidos:
                elegidos.append(c.codigo)
        return sorted((mejores[c] for c in elegidos), key=lambda c: -c.score)

    # -------------------------------------------------- codificación
    def codificar(self, texto: str) -> Resultado:
        t0 = time.perf_counter()
        nota = preprocesar(texto)
        cands = self.candidatos(nota)
        if self.proveedor == "rag":
            res = self._linea_base(nota, cands)
        else:
            res = self._con_llm(nota, cands)
        res.latencia_total_s = time.perf_counter() - t0
        res.nota = nota
        return res

    def _linea_base(self, nota: NotaProcesada, cands: list[Candidato]) -> Resultado:
        """Sin LLM: top-1 del recuperador por cláusula, si supera un umbral de similitud."""
        umbral = self.s.umbral_baseline or self.recuperador.umbral_por_defecto
        codigos: dict[str, CodigoSugerido] = {}
        clausulas = [c for c in nota.clausulas if len(re.sub(r"[^a-záéíóúñ]", "", c.consulta.lower())) >= 3]
        for cl, res in zip(clausulas, self.recuperador.buscar([c.consulta for c in clausulas], 1)):
            if not res or res[0].score < umbral:
                continue
            top = res[0]
            estado = _A_ESTADO[cl.estado]
            previo = codigos.get(top.codigo)
            if previo and (previo.estado == CONFIRMADO or estado != CONFIRMADO):
                continue
            codigos[top.codigo] = CodigoSugerido(
                top.codigo, top.descripcion, estado, cl.texto, cl.inicio, cl.fin,
                f"similitud {top.score:.2f}", True, cl.estado, None, False, top.codigo)
        return Resultado("rag", f"rag-{self.recuperador.backend}", list(codigos.values()), cands, 0.0)

    def _con_llm(self, nota: NotaProcesada, cands: list[Candidato]) -> Resultado:
        usuario = construir_usuario(nota.original, nota.marcado, abreviaturas_en(nota.original),
                                    [(c.codigo, c.descripcion) for c in cands])
        advertencias: list[str] = []
        cliente = self.cliente()
        try:
            r = cliente.generar_json(SISTEMA, usuario, ESQUEMA)
        except ErrorLLM:
            raise
        except Exception as e:  # errores de red / API
            raise ErrorLLM(f"{self.proveedor}: {e}") from e

        en_cands = {c.codigo for c in cands}
        salida: dict[str, CodigoSugerido] = {}
        for item in r.datos.get("codigos", []) or []:
            if not isinstance(item, dict):
                continue
            original = str(item.get("codigo", "")).strip()
            cod_norm = normalizar_codigo(original)
            if cod_norm[:1] in {"V", "W", "X", "Y"}:
                advertencias.append(f"{original}: causa externa descartada")
                continue
            codigo = self.catalogo.validar(cod_norm)
            if not codigo:
                advertencias.append(f"{original}: no existe en el catálogo CIE-10, descartado")
                continue
            estado_llm = str(item.get("estado", CONFIRMADO)).lower().strip()
            if estado_llm not in (CONFIRMADO, "negado", "sospecha"):
                estado_llm = CONFIRMADO
            # el modelo a veces copia la nota marcada por NegEx: "Niega [NEG fiebre]" -> "Niega fiebre"
            evidencia = re.sub(r"\[(?:NEG|SOSP) ([^\]]*)\]", r"\1", str(item.get("evidencia", ""))).strip()
            span = localizar_evidencia(nota.original, evidencia)
            estado_negex = None
            estado = estado_llm
            ajustado = False
            if span:
                estado_negex = _A_ESTADO[nota.estado_en(*span)]
                if self.negacion == "hibrida" and estado_llm == CONFIRMADO and estado_negex != CONFIRMADO:
                    estado, ajustado = estado_negex, True
            else:
                advertencias.append(f"{codigo}: evidencia no encontrada en la nota")
            sug = CodigoSugerido(
                codigo, self.catalogo.descripcion(codigo), estado, evidencia,
                span[0] if span else None, span[1] if span else None,
                str(item.get("justificacion", "")), codigo in en_cands, estado_negex, estado_llm,
                ajustado, original)
            previo = salida.get(codigo)
            if previo is None or (previo.estado != CONFIRMADO and estado == CONFIRMADO):
                salida[codigo] = sug
        self._agregar_negados_negex(nota, salida)
        return Resultado(r.proveedor, r.modelo, list(salida.values()), cands, 0.0, r.latencia_s,
                         r.tokens_entrada, r.tokens_salida, advertencias, respuesta_cruda=r.texto)

    def _agregar_negados_negex(self, nota: NotaProcesada, salida: dict) -> None:
        """Muestra como negadas las menciones que NegEx detectó y el LLM no reportó.

        Solo informa al codificador: nunca agrega códigos confirmados, así que no cambia
        las métricas de precisión/recall.
        """
        cubiertos = [(c.inicio, c.fin) for c in salida.values() if c.inicio is not None]
        for o in nota.oraciones:
            for cl in o.clausulas:
                if cl.estado == AFIRMADO:
                    continue
                if any(a < cl.fin and cl.inicio < b for a, b in cubiertos):
                    continue
                # consulta sin los disparadores ("niega fiebre" -> "fiebre")
                termino = " ".join(t.texto for t in o.tokens
                                   if cl.inicio <= t.inicio and t.fin <= cl.fin and not t.es_disparador
                                   and t.estado != AFIRMADO)
                if len(re.sub(r"[^a-záéíóúñ]", "", termino.lower())) < 4:
                    continue
                consulta, _ = a_consulta(termino)
                res = self.recuperador.buscar([consulta or termino], 1)
                if not res or not res[0]:
                    continue
                top = res[0][0]
                if top.codigo in salida:
                    continue
                estado = _A_ESTADO[cl.estado]
                salida[top.codigo] = CodigoSugerido(
                    top.codigo, top.descripcion, estado, cl.texto, cl.inicio, cl.fin,
                    "Detectado por NegEx; código sugerido por el recuperador", True,
                    estado, None, False, top.codigo)
                cubiertos.append((cl.inicio, cl.fin))
