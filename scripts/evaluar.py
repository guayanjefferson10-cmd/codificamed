"""Evalúa CodificaMed sobre un conjunto de casos con códigos conocidos.

Ejemplos:
    python scripts/evaluar.py --provider ollama
    python scripts/evaluar.py --provider anthropic
    python scripts/evaluar.py --provider rag                      # línea base sin LLM
    python scripts/evaluar.py --provider ollama --dataset data/codiesp_test.jsonl --limite 50
    python scripts/evaluar.py --provider ollama --negacion llm    # ablación: sin corrección NegEx

Genera results/<nombre>/predicciones.jsonl y results/<nombre>/metricas.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codificamed.config import DATA_DIR, PROVIDERS, RESULTS_DIR, get_settings  # noqa: E402
from codificamed.evaluacion import cargar_casos, evaluar  # noqa: E402
from codificamed.pipeline import CodificaMed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", choices=PROVIDERS, help="por defecto, LLM_PROVIDER de .env")
    ap.add_argument("--modelo", help="sobrescribe el modelo del proveedor")
    ap.add_argument("--dataset", default=str(DATA_DIR / "casos_ficticios.jsonl"))
    ap.add_argument("--limite", type=int)
    ap.add_argument("--backend", choices=["hibrido", "e5", "tfidf"], help="backend de embeddings")
    ap.add_argument("--negacion", choices=["hibrida", "llm"], help="política de negación")
    ap.add_argument("--nombre", help="nombre de la carpeta de resultados")
    args = ap.parse_args()

    cambios = {}
    if args.provider:
        cambios["llm_provider"] = args.provider
    if args.backend:
        cambios["embedding_backend"] = args.backend
    s = get_settings(**cambios)
    if args.modelo:
        campo = {"ollama": "ollama_model", "anthropic": "anthropic_model", "openai": "openai_model"}
        if s.llm_provider in campo:
            s = s.con(**{campo[s.llm_provider]: args.modelo})

    casos = cargar_casos(args.dataset, args.limite)
    pipe = CodificaMed(s, negacion=args.negacion)
    nombre = args.nombre or f"{Path(args.dataset).stem}__{s.llm_provider}__{s.modelo_activo.replace(':', '-').replace('/', '-')}"
    if args.negacion:
        nombre += f"__neg-{args.negacion}"
    salida = RESULTS_DIR / nombre
    salida.mkdir(parents=True, exist_ok=True)

    print(f"Evaluando {len(casos)} casos | proveedor={s.llm_provider} modelo={s.modelo_activo} "
          f"| embeddings={s.embedding_backend} | negación={pipe.negacion}")
    preds = []
    t0 = time.time()
    with open(salida / "predicciones.jsonl", "w", encoding="utf-8") as f:
        for i, caso in enumerate(casos, 1):
            try:
                pred = pipe.codificar(caso["texto"]).a_dict()
            except Exception as e:  # se registra y se sigue con el siguiente caso
                pred = {"error": str(e)}
            pred["id"] = caso.get("id")
            preds.append(pred)
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
            conf = [c["codigo"] for c in pred.get("codigos", []) if c["estado"] == "confirmado"]
            estado = f"ERROR: {pred['error'][:80]}" if pred.get("error") else f"{conf}"
            print(f"  [{i:>3}/{len(casos)}] {caso.get('id')}: esperado={caso['codigos']} -> {estado}")

    m = evaluar(casos, preds)
    m.update({"proveedor": s.llm_provider, "modelo": s.modelo_activo, "dataset": Path(args.dataset).name,
              "embeddings": s.embedding_backend, "politica_negacion": pipe.negacion if s.llm_provider != "rag" else "n/a",
              "duracion_total_s": round(time.time() - t0, 1)})
    if s.precio_entrada_mtok or s.precio_salida_mtok:
        m["costo_estimado_usd"] = round(m["tokens_entrada_total"] / 1e6 * s.precio_entrada_mtok
                                        + m["tokens_salida_total"] / 1e6 * s.precio_salida_mtok, 4)
    (salida / "metricas.json").write_text(json.dumps(m, ensure_ascii=False, indent=2), encoding="utf-8")

    ex, cat, neg = m["exacto"], m["categoria"], m["negacion"]
    print("\n=== Resultados ===")
    print(f"Código exacto : P={ex['precision']:.3f} R={ex['recall']:.3f} F1={ex['f1']:.3f}")
    print(f"Categoría (3c): P={cat['precision']:.3f} R={cat['recall']:.3f} F1={cat['f1']:.3f}")
    if neg["menciones_negadas"]:
        print(f"Negación      : {neg['confirmadas_por_error']}/{neg['menciones_negadas']} negados "
              f"confirmados por error (exactitud {neg['exactitud']:.3f})")
    print(f"Latencia media: {m['latencia_media_s']} s | fallos: {m['fallos']}")
    print(f"Guardado en {salida}")


if __name__ == "__main__":
    main()
