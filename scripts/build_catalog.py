"""Construye data/cie10_es.csv a partir del catálogo oficial CIE-10 en español.

Fuente: lista tabular CIE-10 (OMS, 2008) publicada por MINSAL/DEIS Chile, redistribuida en
https://github.com/tamycova/icd-cie-codes-chile (diagnoses/es/original.xls).

Se conservan únicamente categorías (3 caracteres, p. ej. "E11") y subcategorías
(4 caracteres, p. ej. "E11.9"), que es el nivel que usa el MSP del Ecuador (RDACAA).
Para cada subcategoría se arma un "texto de búsqueda" que antepone la descripción de su
categoría padre, porque muchas subcategorías por sí solas no tienen sentido
("E11.9 Sin mención de complicación").

Uso:
    python scripts/build_catalog.py [ruta_xls]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "cie10_minsal_2008.xls"
OUT = ROOT / "data" / "cie10_es.csv"

CODE_RE = re.compile(r"^[A-Z]\d\d(\.\d)?$")


def main() -> None:
    raw = pd.read_excel(SRC, sheet_name="2008", dtype=str)
    raw["codigo"] = raw["codigo"].str.strip()
    raw["descriptor"] = raw["descriptor"].str.strip().str.replace(r"\s+", " ", regex=True)

    # Capítulo vigente para cada fila (las filas "Cap.XX" preceden a sus códigos)
    capitulo = None
    caps = []
    for cod, desc in zip(raw["codigo"], raw["descriptor"]):
        if isinstance(cod, str) and cod.startswith("Cap."):
            capitulo = desc
        caps.append(capitulo)
    raw["capitulo"] = caps

    df = raw[raw["codigo"].fillna("").str.match(CODE_RE)].copy()
    df["categoria"] = df["codigo"].str[:3]
    cat_desc = df[df["codigo"].str.len() == 3].set_index("codigo")["descriptor"].to_dict()
    df["descripcion_categoria"] = df["categoria"].map(cat_desc)
    df["solo_secundario"] = df["cod_sec"].fillna("").eq("S")

    def texto(row) -> str:
        if len(row.codigo) == 3 or not row.descripcion_categoria:
            return row.descriptor
        return f"{row.descripcion_categoria}: {row.descriptor}"

    df["texto_busqueda"] = df.apply(texto, axis=1)
    out = df.rename(columns={"descriptor": "descripcion"})[
        ["codigo", "descripcion", "categoria", "descripcion_categoria",
         "capitulo", "solo_secundario", "texto_busqueda"]
    ]
    out.to_csv(OUT, index=False)
    print(f"{len(out)} códigos -> {OUT}")


if __name__ == "__main__":
    main()
