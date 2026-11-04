"""Convierte el corpus CodiEsp (CLEF eHealth 2020) al formato JSONL de evaluación.

1. Descarga el corpus desde Zenodo: https://zenodo.org/records/3837305
   (archivo codiesp.zip) y descomprímelo, por ejemplo en data/codiesp/
2. Ejecuta:
       python scripts/preparar_codiesp.py --dir data/codiesp --split test --n 100

Notas:
- CodiEsp está anotado con CIE-10-ES (modificación clínica española, hasta 7 caracteres,
  p. ej. "s72.001a"). Este proyecto usa la CIE-10 de la OMS (4 caracteres), así que cada
  código se trunca a su subcategoría ("S72.0") y, si no existe, a su categoría ("S72").
  Por eso en CodiEsp la métrica principal es la de CATEGORÍA (3 caracteres).
- Se usan solo los diagnósticos (archivo <split>D.tsv); se excluyen las causas externas.
- Los casos de CodiEsp son casos clínicos publicados en revistas, no historias reales de
  pacientes, por lo que cumplen la restricción de la tarea.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codificamed.catalog import cargar_catalogo, normalizar_codigo  # noqa: E402
from codificamed.config import DATA_DIR  # noqa: E402


def buscar_base(d: Path, split: str) -> Path:
    for cand in [d, *d.rglob(split)]:
        base = cand if cand.name == split else cand / split
        if (base / "text_files").is_dir():
            return base
    raise SystemExit(f"No se encontró la carpeta '{split}/text_files' dentro de {d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="carpeta donde se descomprimió CodiEsp")
    ap.add_argument("--split", default="test", choices=["train", "dev", "test"])
    ap.add_argument("--n", type=int, default=100, help="número de casos (muestra aleatoria)")
    ap.add_argument("--max-chars", type=int, default=3500, help="descarta casos más largos")
    ap.add_argument("--semilla", type=int, default=42)
    args = ap.parse_args()

    base = buscar_base(Path(args.dir), args.split)
    tsv = base / f"{args.split}D.tsv"
    cat = cargar_catalogo()
    codigos: dict[str, list[str]] = defaultdict(list)
    sin_mapeo = 0
    with open(tsv, encoding="utf-8") as f:
        for fila in csv.reader(f, delimiter="\t"):
            if len(fila) < 2:
                continue
            doc, cod = fila[0].strip(), fila[1].strip()
            if normalizar_codigo(cod)[:1] in "VWXY":
                continue
            v = cat.validar(cod)
            if v:
                codigos[doc].append(v)
            else:
                sin_mapeo += 1

    casos = []
    for doc, cods in codigos.items():
        ruta = base / "text_files" / f"{doc}.txt"
        if not ruta.exists():
            continue
        texto = ruta.read_text(encoding="utf-8").strip()
        if len(texto) > args.max_chars:
            continue
        casos.append({"id": doc, "texto": texto, "codigos": sorted(set(cods)),
                      "opcionales": [], "negados": []})
    random.Random(args.semilla).shuffle(casos)
    casos = casos[: args.n]
    out = DATA_DIR / f"codiesp_{args.split}.jsonl"
    with open(out, "w", encoding="utf-8") as f:
        for c in casos:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(casos)} casos -> {out} (códigos sin equivalente OMS descartados: {sin_mapeo})")


if __name__ == "__main__":
    main()
