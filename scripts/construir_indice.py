"""Precalcula el índice de embeddings del catálogo CIE-10 (se guarda en .cache/).

    python scripts/construir_indice.py              # backend de .env (e5 por defecto)
    python scripts/construir_indice.py --backend tfidf

La primera vez descarga el modelo intfloat/multilingual-e5-small (~470 MB) desde Hugging Face.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codificamed.config import get_settings  # noqa: E402
from codificamed.recuperador import Recuperador  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=["hibrido", "e5", "tfidf"])
a = ap.parse_args()
s = get_settings(**({"embedding_backend": a.backend} if a.backend else {}))
t = time.time()
r = Recuperador(s)
print(f"Índice {s.embedding_backend} listo: {len(r.catalogo)} códigos en {time.time() - t:.1f} s")
for q in ["hipertensión arterial", "diabetes mellitus tipo 2", "neumonía adquirida en la comunidad"]:
    print(f"  {q!r:40} -> {[(c.codigo, round(c.score, 2)) for c in r.buscar([q], 3)[0]]}")
