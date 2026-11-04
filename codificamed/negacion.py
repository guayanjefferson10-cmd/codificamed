"""Detección de negación e incertidumbre basada en reglas (NegEx adaptado al español).

Algoritmo (Chapman et al., 2001, adaptado):
  1. Se buscan disparadores en la oración ("niega", "se descarta", "sin", "probable"...).
  2. Un disparador PRE niega los tokens que le siguen; uno POST niega los que le preceden.
  3. El alcance termina en un terminador ("pero", "aunque"...), en ':' o ';', en otro
     disparador o tras MAX_ALCANCE tokens.
  4. Cada token queda etiquetado como 'afirmado', 'negado' o 'sospecha'.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

MAX_ALCANCE = 6

AFIRMADO, NEGADO, SOSPECHA = "afirmado", "negado", "sospecha"

# (frase, tipo, dirección). Se ordenan de más larga a más corta al compilar.
DISPARADORES: list[tuple[str, str, str]] = [
    # Negación previa al término
    ("se descarta", NEGADO, "pre"), ("se descartó", NEGADO, "pre"), ("se decarta", NEGADO, "pre"),
    ("descarta", NEGADO, "pre"), ("descartándose", NEGADO, "pre"),
    ("niega", NEGADO, "pre"), ("negó", NEGADO, "pre"), ("niegan", NEGADO, "pre"),
    ("no presenta", NEGADO, "pre"), ("no refiere", NEGADO, "pre"), ("no se observa", NEGADO, "pre"),
    ("no se observan", NEGADO, "pre"), ("no se evidencia", NEGADO, "pre"), ("no evidencia", NEGADO, "pre"),
    ("no hay", NEGADO, "pre"), ("no tiene", NEGADO, "pre"), ("no", NEGADO, "pre"),
    ("sin evidencia de", NEGADO, "pre"), ("sin signos de", NEGADO, "pre"), ("sin datos de", NEGADO, "pre"),
    ("sin", NEGADO, "pre"), ("ausencia de", NEGADO, "pre"), ("libre de", NEGADO, "pre"),
    ("negativo para", NEGADO, "pre"), ("negativa para", NEGADO, "pre"),
    ("ni", NEGADO, "pre"),
    # Negación posterior al término ("BHCG negativa", "TEP descartado")
    ("descartado", NEGADO, "post"), ("descartada", NEGADO, "post"),
    ("negativo", NEGADO, "post"), ("negativa", NEGADO, "post"),
    ("negativos", NEGADO, "post"), ("negativas", NEGADO, "post"),
    ("ausente", NEGADO, "post"),
    # Incertidumbre (no se codifica como confirmado)
    ("a descartar", SOSPECHA, "pre"), ("descartar", SOSPECHA, "pre"), ("d/c", SOSPECHA, "pre"),
    ("sospecha de", SOSPECHA, "pre"), ("probable", SOSPECHA, "pre"), ("posible", SOSPECHA, "pre"),
    ("presunto", SOSPECHA, "pre"), ("presunta", SOSPECHA, "pre"),
    ("compatible con", SOSPECHA, "pre"), ("vs", SOSPECHA, "pre"),
]

# Palabras que cortan el alcance de un disparador
TERMINADORES = {"pero", "aunque", "sin embargo", "excepto", "salvo", "mas", "dx", "diagnóstico",
                "idx", "presenta", "refiere", "acude", "con"}

# Palabras tras las cuales "no" NO es negación clínica ("no especificado", "no compresible")
FALSOS_NO = {"especificado", "especificada", "especificados", "compresible", "insulinodependiente",
             "obstante", "solo", "sólo", "mejora", "cede"}

_TOKEN_RE = re.compile(r"\w+(?:[/+]\w*)?|[^\w\s]", re.UNICODE)


def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


@dataclass
class Token:
    texto: str
    inicio: int
    fin: int
    estado: str = AFIRMADO
    es_disparador: bool = False


@dataclass
class Disparador:
    frase: str
    tipo: str
    direccion: str
    inicio: int  # índice de token
    fin: int     # índice de token (exclusivo)


def _compilar():
    tabla = []
    for frase, tipo, dirc in DISPARADORES:
        partes = tuple(_sin_tildes(p).lower() for p in _TOKEN_RE.findall(frase))
        tabla.append((partes, frase, tipo, dirc))
    tabla.sort(key=lambda x: -len(x[0]))
    return tabla


_TABLA = _compilar()
_TERM = {_sin_tildes(t) for t in TERMINADORES}


def tokenizar(texto: str, offset: int = 0) -> list[Token]:
    return [Token(m.group(), m.start() + offset, m.end() + offset) for m in _TOKEN_RE.finditer(texto)]


def analizar(texto: str, offset: int = 0) -> tuple[list[Token], list[Disparador]]:
    """Etiqueta cada token de `texto` (una oración) como afirmado/negado/sospecha."""
    toks = tokenizar(texto, offset)
    norm = [_sin_tildes(t.texto).lower() for t in toks]
    disps: list[Disparador] = []
    i = 0
    while i < len(toks):
        hallado = None
        for partes, frase, tipo, dirc in _TABLA:
            n = len(partes)
            if tuple(norm[i:i + n]) == partes:
                if partes == ("no",) and i + 1 < len(norm) and norm[i + 1] in FALSOS_NO:
                    continue
                hallado = Disparador(frase, tipo, dirc, i, i + n)
                break
        if hallado:
            disps.append(hallado)
            for k in range(hallado.inicio, hallado.fin):
                toks[k].es_disparador = True
            i = hallado.fin
        else:
            i += 1

    def corta(k: int) -> bool:
        return norm[k] in {":", ";", ".", "(", ")"} or norm[k] in _TERM or toks[k].es_disparador

    for d in disps:
        contados = 0
        if d.direccion == "pre":
            k = d.fin
            while k < len(toks) and contados < MAX_ALCANCE and not corta(k):
                if norm[k] != ",":
                    toks[k].estado = d.tipo
                    contados += 1
                k += 1
        else:
            k = d.inicio - 1
            while k >= 0 and contados < MAX_ALCANCE - 2 and not corta(k) and norm[k] != ",":
                toks[k].estado = d.tipo
                contados += 1
                k -= 1
    return toks, disps


STOP = {"de", "del", "la", "el", "los", "las", "y", "e", "o", "a", "en", "con", "por", "para",
        "al", "un", "una", "se", ",", ".", ":", ";", "(", ")"}


# Palabras que no son el "núcleo" de una mención (exámenes, relleno)
NO_NUCLEO = STOP | {"rx", "radiografía", "eco", "ecografía", "tac", "ekg", "ecg", "test", "prueba",
                    "examen", "signos", "datos", "evidencia", "dx", "idx", "se", "paciente", "pcte"}


def estado_de_span(tokens: list[Token], inicio: int, fin: int) -> str:
    """Estado del núcleo de la mención: primer token de contenido dentro de [inicio, fin).

    En español el término clínico suele ir primero ("DM2 sin complicaciones" -> DM2 afirmada;
    "se descarta neumonía" -> neumonía negada).
    """
    dentro = [t for t in tokens if t.inicio >= inicio and t.fin <= fin and not t.es_disparador
              and re.match(r"\w", t.texto)]
    for t in dentro:
        if t.texto.lower() not in NO_NUCLEO and not t.texto.isdigit():
            return t.estado
    for t in dentro:
        if t.estado != AFIRMADO:
            return t.estado
    return AFIRMADO


def marcar(tokens: list[Token], texto: str, offset: int = 0) -> str:
    """Devuelve el texto con los alcances marcados: 'niega [NEG fiebre]'. Se usa en el prompt."""
    etiqueta = {NEGADO: "NEG", SOSPECHA: "SOSP"}
    out, pos, abierto = [], 0, AFIRMADO
    for t in tokens:
        est = AFIRMADO if t.es_disparador else t.estado
        a, b = t.inicio - offset, t.fin - offset
        if est != abierto:
            if abierto != AFIRMADO:
                out.append("]")
            out.append(texto[pos:a])
            pos = a
            if est != AFIRMADO:
                out.append(f"[{etiqueta[est]} ")
            abierto = est
        out.append(texto[pos:b])
        pos = b
    if abierto != AFIRMADO:
        out.append("]")
    out.append(texto[pos:])
    return "".join(out)
