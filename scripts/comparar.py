"""Compara los resultados de todas las evaluaciones guardadas en results/.

    python scripts/comparar.py                 # todas las corridas
    python scripts/comparar.py --dataset casos_ficticios

Genera results/comparacion.md (tabla lista para pegar en el informe) y
results/comparacion.png (gráfico de barras P/R/F1 por proveedor).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from codificamed.config import RESULTS_DIR  # noqa: E402

# Color fijo por proveedor (la identidad no cambia aunque falte alguna corrida)
COLORES = {"rag": "#8a8984", "ollama": "#2a78d6", "anthropic": "#eb6834", "openai": "#1baf7a"}
ETIQUETAS = {"rag": "Línea base (solo RAG)", "ollama": "Local (Ollama)",
             "anthropic": "API (Claude)", "openai": "API (OpenAI)"}


def cargar(dataset: str | None) -> list[dict]:
    corridas = []
    for p in sorted(RESULTS_DIR.glob("*/metricas.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        if dataset and not m.get("dataset", "").startswith(dataset):
            continue
        m["carpeta"] = p.parent.name
        corridas.append(m)
    orden = list(COLORES)
    return sorted(corridas, key=lambda m: (m.get("dataset", ""), orden.index(m["proveedor"])
                                           if m["proveedor"] in orden else 9))


def tabla_md(corridas: list[dict]) -> str:
    filas = ["| Dataset | Proveedor | Modelo | Negación | P (exacto) | R (exacto) | F1 (exacto) | "
             "F1 (categoría) | Negados confirmados | Latencia media (s) | Tokens entrada/salida |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in corridas:
        ex, cat, neg = m["exacto"], m["categoria"], m["negacion"]
        negtxt = f"{neg['confirmadas_por_error']}/{neg['menciones_negadas']}" if neg["menciones_negadas"] else "—"
        filas.append(
            f"| {m['dataset']} | {ETIQUETAS.get(m['proveedor'], m['proveedor'])} | {m['modelo']} | "
            f"{m.get('politica_negacion', '')} | {ex['precision']:.3f} | {ex['recall']:.3f} | "
            f"**{ex['f1']:.3f}** | {cat['f1']:.3f} | {negtxt} | {m['latencia_media_s']} | "
            f"{m['tokens_entrada_total']}/{m['tokens_salida_total']} |")
    return "\n".join(filas)


def grafico(corridas: list[dict], ruta: Path, nivel: str = "exacto") -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not corridas:
        return None
    metricas = [("precision", "Precisión"), ("recall", "Recall"), ("f1", "F1")]
    n = len(corridas)
    ancho = 0.8 / n
    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=150)
    fig.patch.set_facecolor("#fcfcfb")
    ax.set_facecolor("#fcfcfb")
    for i, m in enumerate(corridas):
        xs = [j + (i - (n - 1) / 2) * ancho for j in range(len(metricas))]
        vals = [m[nivel][k] for k, _ in metricas]
        color = COLORES.get(m["proveedor"], "#6250d6")
        etiqueta = f"{ETIQUETAS.get(m['proveedor'], m['proveedor'])} · {m['modelo']}"
        if m.get("politica_negacion") == "llm" and m["proveedor"] != "rag":
            etiqueta += " (sin NegEx)"
        barras = ax.bar(xs, vals, width=ancho * 0.92, color=color, label=etiqueta, zorder=3)
        for b, v in zip(barras, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center",
                    va="bottom", fontsize=8, color="#52514e")
    ax.set_xticks(range(len(metricas)), [t for _, t in metricas], color="#0b0b0b")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Valor (0–1)", color="#52514e")
    ax.grid(axis="y", color="#e6e5e0", linewidth=0.8, zorder=0)
    for lado in ("top", "right", "left"):
        ax.spines[lado].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(colors="#52514e", length=0)
    titulo = "código exacto" if nivel == "exacto" else "categoría (3 caracteres)"
    ax.set_title(f"CodificaMed: métricas por proveedor ({titulo})", loc="left", fontsize=11,
                 color="#0b0b0b")
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0, -0.08), ncol=2)
    fig.tight_layout()
    fig.savefig(ruta, facecolor=fig.get_facecolor())
    plt.close(fig)
    return ruta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="prefijo del dataset a incluir (p. ej. casos_ficticios)")
    args = ap.parse_args()
    corridas = cargar(args.dataset)
    if not corridas:
        print("No hay resultados. Ejecuta primero scripts/evaluar.py")
        return
    md = tabla_md(corridas)
    (RESULTS_DIR / "comparacion.md").write_text("# Comparación de proveedores\n\n" + md + "\n",
                                               encoding="utf-8")
    grafico(corridas, RESULTS_DIR / "comparacion.png")
    grafico(corridas, RESULTS_DIR / "comparacion_categoria.png", "categoria")
    print(md)
    print(f"\nTabla: {RESULTS_DIR / 'comparacion.md'}\nGráficos: {RESULTS_DIR / 'comparacion.png'}")


if __name__ == "__main__":
    main()
