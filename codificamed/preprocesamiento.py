"""Preprocesamiento del texto clínico.

Pasos:
  1. Limpieza: espacios, comillas y caracteres de control (se conservan los offsets).
  2. Segmentación en oraciones y en cláusulas (unidad de "mención" para el recuperador).
  3. Detección de negación/incertidumbre por oración (NegEx-es).
  4. Normalización de cada cláusula para la búsqueda: expansión de abreviaturas,
     corrección ortográfica ligera contra el vocabulario del catálogo y retiro de relleno
     ("paciente de 58 años con...").

El texto original nunca se modifica: la evidencia se muestra siempre sobre la nota original.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from functools import lru_cache

from rapidfuzz import fuzz, process

from .abreviaturas import ABREVIATURAS, REGLAS_CONTEXTO
from .negacion import AFIRMADO, Token, analizar, estado_de_span, marcar

# ---------------------------------------------------------------- segmentación
# Punto que NO termina oración: entre dígitos (38.2, 120.000) o tras abreviaturas comunes
_NO_FIN = re.compile(r"(?:\d\.\d|\b(?:Dr|Dra|Sr|Sra|aprox|H|p\.ej|vs|cm|mg|ml)\.)", re.I)
_FIN_ORACION = re.compile(r"[.;\n]+")
_SEP_CLAUSULA = re.compile(r"\s*[,:;]\s*|\s+(?:y|e)\s+(?=\S)")

_ABBR_ORDEN = sorted(ABREVIATURAS, key=len, reverse=True)
_ABBR_RE = re.compile(
    r"(?<![\w])(" + "|".join(re.escape(a) for a in _ABBR_ORDEN) + r")(?![\w])",
    re.IGNORECASE,
)
_ABBR_LOWER = {k.lower(): v for k, v in ABREVIATURAS.items()}

_RELLENO = re.compile(
    r"\b(?:pcte|pte|paciente|varón|varon|hombre|mujer|niño|niña|lactante|adulto|adulta|mayor|"
    r"masculino|femenina|femenino|de\s+\d+\s*(?:años|a|meses|días|semanas)|\d+\s*a\b|"
    r"acude(?:\s+a\s+control)?|ingresa|traíd[oa]|refiere|presenta|con\s+antecedentes?\s+de|"
    r"antecedentes?\s+de|(?:diagnóstico|impresión\s+diagnóstica|dx|idx)(?:\s+de)?)\b",
    re.IGNORECASE,
)


def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


@dataclass
class Clausula:
    texto: str            # texto original de la cláusula
    inicio: int           # offset en la nota
    fin: int
    estado: str           # afirmado | negado | sospecha (según NegEx)
    consulta: str = ""    # texto normalizado para el recuperador
    correcciones: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Oracion:
    texto: str
    inicio: int
    fin: int
    tokens: list[Token]
    disparadores: list
    clausulas: list[Clausula]


@dataclass
class NotaProcesada:
    original: str
    oraciones: list[Oracion]
    texto_expandido: str

    @property
    def marcado(self) -> list[str]:
        """Oraciones con los alcances de NegEx marcados, para el prompt."""
        return [marcar(o.tokens, o.texto, o.inicio) for o in self.oraciones]

    @property
    def clausulas(self) -> list[Clausula]:
        return [c for o in self.oraciones for c in o.clausulas]

    def estado_en(self, inicio: int, fin: int) -> str:
        for o in self.oraciones:
            if o.inicio <= inicio < o.fin:
                return estado_de_span(o.tokens, inicio, fin)
        return AFIRMADO

    def oracion_de(self, inicio: int) -> Oracion | None:
        for o in self.oraciones:
            if o.inicio <= inicio < o.fin:
                return o
        return None


def segmentar_oraciones(texto: str) -> list[tuple[int, int]]:
    protegidos = {m.start() + m.group().rindex(".") for m in _NO_FIN.finditer(texto)}
    spans, ini = [], 0
    for m in _FIN_ORACION.finditer(texto):
        if m.start() in protegidos:
            continue
        if texto[ini:m.start()].strip():
            spans.append((ini, m.start()))
        ini = m.end()
    if texto[ini:].strip():
        spans.append((ini, len(texto)))
    # recortar espacios
    out = []
    for a, b in spans:
        while a < b and texto[a].isspace():
            a += 1
        while b > a and texto[b - 1].isspace():
            b -= 1
        out.append((a, b))
    return out


def segmentar_clausulas(texto: str, offset: int) -> list[tuple[int, int]]:
    spans, ini = [], 0
    # no cortar comas dentro de números (120,000) ni de paréntesis
    prof = 0
    cortes = []
    for m in _SEP_CLAUSULA.finditer(texto):
        prof = texto[:m.start()].count("(") - texto[:m.start()].count(")")
        if prof > 0:
            continue
        if m.group().strip() == "," and m.start() > 0 and texto[m.start() - 1].isdigit() \
                and m.end() < len(texto) and texto[m.end()].isdigit():
            continue
        cortes.append(m)
    for m in cortes:
        if texto[ini:m.start()].strip():
            spans.append((ini + offset, m.start() + offset))
        ini = m.end()
    if texto[ini:].strip():
        spans.append((ini + offset, len(texto) + offset))
    return spans


# ---------------------------------------------------------------- normalización
def expandir_abreviaturas(texto: str) -> str:
    for sigla, patron_sig, expansion in REGLAS_CONTEXTO:
        texto = re.sub(rf"(?<![\w]){re.escape(sigla)}(?={patron_sig})", expansion, texto)

    def rep(m: re.Match) -> str:
        return _ABBR_LOWER.get(m.group(1).lower(), m.group(1))

    texto = _ABBR_RE.sub(rep, texto)
    texto = re.sub(r"\b(\d+)\s*a\b", r"\1 años", texto)
    return texto


@lru_cache(maxsize=1)
def _vocabulario() -> tuple[frozenset, list[str]]:
    """Vocabulario médico: palabras del catálogo + expansiones + palabras clínicas comunes."""
    from .catalog import cargar_catalogo

    palabras: set[str] = set()
    for t in cargar_catalogo().textos:
        palabras.update(w for w in re.findall(r"[a-záéíóúñü]+", t.lower()) if len(w) >= 4)
    for v in ABREVIATURAS.values():
        palabras.update(re.findall(r"[a-záéíóúñü]+", v.lower()))
    palabras.update(
        "descarta descartado descartada niega refiere presenta evolución antecedente antecedentes "
        "tratamiento control paciente dolor fiebre tos disnea días semanas meses años mecánico "
        "osteoartrosis gonartrosis coxartrosis bilateral derecho derecha izquierdo izquierda "
        "positivo positiva negativo negativa crónico crónica agudo aguda leve moderado moderada "
        "severo severa grave secundaria secundario asociado asociada reciente diagnóstico".split()
    )
    sin = frozenset(_sin_tildes(p) for p in palabras)
    prefijos = frozenset(_sin_tildes(p)[:k] for p in palabras for k in range(5, len(p) + 1))
    return sin, prefijos, sorted(palabras)


def corregir_ortografia(texto: str) -> tuple[str, list[tuple[str, str]]]:
    """Corrige palabras largas fuera del vocabulario médico (p. ej. 'gonartosis' -> 'gonartrosis')."""
    vocab_sin, prefijos, vocab = _vocabulario()
    cambios = []

    def rep(m: re.Match) -> str:
        w = m.group()
        wl = w.lower()
        ws = _sin_tildes(wl)
        # palabra conocida, o variante de número/género de una conocida ("conocido"/"conocidos")
        if len(wl) < 7 or ws in vocab_sin or ws[:-2] in prefijos:
            return w
        mejor = process.extractOne(wl, vocab, scorer=fuzz.ratio, score_cutoff=88)
        if mejor and mejor[0] != wl and mejor[0][0] == wl[0] and abs(len(mejor[0]) - len(wl)) <= 2:
            cambios.append((w, mejor[0]))
            return mejor[0]
        return w

    return re.sub(r"[A-Za-záéíóúñüÁÉÍÓÚÑÜ]+", rep, texto), cambios


def a_consulta(texto: str, siguiente: str = "") -> tuple[str, list[tuple[str, str]]]:
    # `siguiente` = primer carácter tras la cláusula; resuelve siglas ambiguas ("EDA:")
    t = expandir_abreviaturas(texto + siguiente).removesuffix(siguiente) if siguiente else \
        expandir_abreviaturas(texto)
    t, cambios = corregir_ortografia(t)
    t = _RELLENO.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip(" ,.:;-")
    return t, cambios


def limpiar(texto: str) -> str:
    """Normaliza espacios y comillas sin cambiar la longitud del texto (conserva offsets)."""
    tabla = str.maketrans({"“": '"', "”": '"', "’": "'", "\t": " ", "\r": " ",
                           " ": " "})
    return texto.translate(tabla)


def preprocesar(nota: str) -> NotaProcesada:
    texto = limpiar(nota)
    oraciones = []
    for a, b in segmentar_oraciones(texto):
        frag = texto[a:b]
        toks, disps = analizar(frag, offset=a)
        clausulas = []
        for ca, cb in segmentar_clausulas(frag, a):
            cl_txt = texto[ca:cb]
            sig = texto[cb:cb + 3].strip()[:1]
            consulta, cambios = a_consulta(cl_txt, sig if sig in ":" and sig else "")
            clausulas.append(Clausula(cl_txt, ca, cb, estado_de_span(toks, ca, cb), consulta, cambios))
        oraciones.append(Oracion(frag, a, b, toks, disps, clausulas))
    return NotaProcesada(texto, oraciones, expandir_abreviaturas(texto))
