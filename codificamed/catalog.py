"""Acceso al catálogo CIE-10 en español (data/cie10_es.csv)."""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import pandas as pd

from .config import DATA_DIR

CATALOG_PATH = DATA_DIR / "cie10_es.csv"
SINONIMOS_PATH = DATA_DIR / "sinonimos_cie10.csv"
_CODE_RE = re.compile(r"^([A-Z])(\d{2})\.?(\d)?")


def normalizar_codigo(codigo: str) -> str:
    """'e119' -> 'E11.9', 'i10' -> 'I10', 'S72.001A' (CIE-10-ES) -> 'S72.0'."""
    c = (codigo or "").strip().upper().replace(" ", "")
    m = _CODE_RE.match(c)
    if not m:
        return c
    letra, num, sub = m.groups()
    return f"{letra}{num}.{sub}" if sub else f"{letra}{num}"


def categoria(codigo: str) -> str:
    return normalizar_codigo(codigo)[:3]


@dataclass(frozen=True)
class Entrada:
    codigo: str
    descripcion: str
    categoria: str
    descripcion_categoria: str
    capitulo: str
    texto_busqueda: str


class Catalogo:
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self._idx = {c: i for i, c in enumerate(self.df["codigo"])}

    def __len__(self) -> int:
        return len(self.df)

    def __contains__(self, codigo: str) -> bool:
        return normalizar_codigo(codigo) in self._idx

    def get(self, codigo: str) -> Entrada | None:
        i = self._idx.get(normalizar_codigo(codigo))
        if i is None:
            return None
        r = self.df.iloc[i]
        return Entrada(r.codigo, r.descripcion, r.categoria, r.descripcion_categoria or "",
                       r.capitulo or "", r.texto_busqueda)

    def descripcion(self, codigo: str) -> str:
        """Descripción legible; para subcategorías antepone la categoría
        ("E11.9" -> "Diabetes mellitus no insulinodependiente: Sin mención de complicación")."""
        e = self.get(codigo)
        if not e:
            return ""
        if len(e.codigo) > 3 and e.descripcion_categoria and \
                e.descripcion_categoria.split(",")[0].lower() not in e.descripcion.lower():
            return f"{e.descripcion_categoria}: {e.descripcion}"
        return e.descripcion

    def validar(self, codigo: str) -> str | None:
        """Devuelve el código normalizado si existe en el catálogo.

        Si el modelo devuelve un nivel más fino que no existe (p. ej. 'E11.90'),
        se prueba con la subcategoría y luego con la categoría.
        """
        c = normalizar_codigo(codigo)
        if c in self._idx:
            return c
        if c[:3] in self._idx:
            return c[:3]
        return None

    @property
    def codigos(self) -> list[str]:
        return list(self.df["codigo"])

    @property
    def textos(self) -> list[str]:
        return list(self.df["texto_busqueda"])


def _agregar_sinonimos(df: pd.DataFrame) -> pd.DataFrame:
    """Enriquece el texto de búsqueda con sinónimos clínicos de uso local.

    Un sinónimo definido para una categoría (p. ej. "E11: diabetes mellitus tipo 2") se
    agrega también a todas sus subcategorías (E11.0 ... E11.9).
    """
    if not SINONIMOS_PATH.exists():
        return df
    sin = pd.read_csv(SINONIMOS_PATH, dtype=str).dropna()
    mapa = {normalizar_codigo(c): s.replace(";", ",") for c, s in zip(sin.codigo, sin.sinonimos)}
    df = df.copy()

    def enriquecer(row):
        extra = [mapa[k] for k in (row.categoria, row.codigo) if k in mapa]
        extra = list(dict.fromkeys(extra))
        return f"{row.texto_busqueda} ({'; '.join(extra)})" if extra else row.texto_busqueda

    df["texto_busqueda"] = df.apply(enriquecer, axis=1)
    return df


@lru_cache(maxsize=1)
def cargar_catalogo(excluir_causas_externas: bool = True) -> Catalogo:
    df = pd.read_csv(CATALOG_PATH, dtype=str, keep_default_na=False)
    df = _agregar_sinonimos(df)
    if excluir_causas_externas:
        # Capítulo XX (V01-Y98): causas externas. No son diagnósticos y se excluyen del índice.
        df = df[~df["codigo"].str[0].isin(["V", "W", "X", "Y"])]
    return Catalogo(df)
