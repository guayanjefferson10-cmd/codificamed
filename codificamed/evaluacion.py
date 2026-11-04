"""Métricas de evaluación: precisión, recall y F1 (micro y macro) a dos niveles.

Formato de cada caso (JSONL):
  {"id": "F01", "texto": "...",
   "codigos":    ["I10", "J44.0|J44.1"],   # códigos esperados; "A|B" = cualquiera es válido
   "opcionales": ["Z87.8"],                # si se predicen no son error; si faltan tampoco
   "negados":    ["R50.9"]}                # mencionados pero negados: NO deben confirmarse

Niveles:
  - exacto:    código completo (E11.9)
  - categoria: tres primeros caracteres (E11), estándar en la literatura de CodiEsp
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path

from .catalog import normalizar_codigo


def cargar_casos(ruta: str | Path, limite: int | None = None) -> list[dict]:
    casos = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            if linea.strip():
                casos.append(json.loads(linea))
    return casos[:limite] if limite else casos


def _nivel(c: str, nivel: str) -> str:
    c = normalizar_codigo(c)
    return c[:3] if nivel == "categoria" else c


def emparejar(gold: list[str], pred: list[str], opcionales: list[str] = (), nivel: str = "exacto"):
    """Devuelve (tp, fp, fn, detalle) para un caso."""
    grupos = [{_nivel(a, nivel) for a in g.split("|")} for g in gold]
    pred_n = list(dict.fromkeys(_nivel(p, nivel) for p in pred))
    opc = {_nivel(o, nivel) for o in opcionales}
    usados: set[str] = set()
    tp_items, fn_items = [], []
    for g, original in zip(grupos, gold):
        hit = next((p for p in pred_n if p in g and p not in usados), None)
        if hit:
            usados.add(hit)
            tp_items.append(hit)
        else:
            fn_items.append(original)
    todos_gold = set().union(*grupos) if grupos else set()
    fp_items = [p for p in pred_n if p not in usados and p not in opc and p not in todos_gold]
    return len(tp_items), len(fp_items), len(fn_items), {"tp": tp_items, "fp": fp_items, "fn": fn_items}


def prf(tp: int, fp: int, fn: int) -> dict:
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": fp, "fn": fn}


@dataclass
class Acumulador:
    nivel: str
    tp: int = 0
    fp: int = 0
    fn: int = 0
    f1_casos: list[float] = field(default_factory=list)

    def agregar(self, gold, pred, opcionales):
        tp, fp, fn, det = emparejar(gold, pred, opcionales, self.nivel)
        self.tp, self.fp, self.fn = self.tp + tp, self.fp + fp, self.fn + fn
        self.f1_casos.append(prf(tp, fp, fn)["f1"])
        return det

    def resumen(self) -> dict:
        d = prf(self.tp, self.fp, self.fn)
        d["f1_macro"] = round(statistics.mean(self.f1_casos), 4) if self.f1_casos else 0.0
        return d


def evaluar(casos: list[dict], predicciones: list[dict]) -> dict:
    """`predicciones[i]` = salida de Resultado.a_dict() (o {"error": ...}) del caso i."""
    exacto, categoria = Acumulador("exacto"), Acumulador("categoria")
    neg_total = neg_error = 0
    lat, lat_llm, tin, tout = [], [], 0, 0
    evid_ok = evid_total = ajustes = fallos = 0
    detalle = []
    for caso, pred in zip(casos, predicciones):
        if pred.get("error"):
            fallos += 1
            confirmados = []
        else:
            confirmados = [c["codigo"] for c in pred["codigos"] if c["estado"] == "confirmado"]
            lat.append(pred["latencia_total_s"])
            lat_llm.append(pred.get("latencia_llm_s", 0))
            tin += pred.get("tokens_entrada", 0)
            tout += pred.get("tokens_salida", 0)
            for c in pred["codigos"]:
                evid_total += 1
                evid_ok += c.get("inicio") is not None
                ajustes += bool(c.get("ajustado_por_negex"))
        opc = caso.get("opcionales", [])
        d_ex = exacto.agregar(caso["codigos"], confirmados, opc)
        d_cat = categoria.agregar(caso["codigos"], confirmados, opc)
        negados = [normalizar_codigo(n) for n in caso.get("negados", [])]
        errores_neg = [n for n in negados if n in {normalizar_codigo(c) for c in confirmados}]
        neg_total += len(negados)
        neg_error += len(errores_neg)
        detalle.append({"id": caso.get("id"), "gold": caso["codigos"], "pred": confirmados,
                        "exacto": d_ex, "categoria": d_cat, "negados_confirmados": errores_neg,
                        "error": pred.get("error")})
    n = len(casos)
    return {
        "casos": n,
        "fallos": fallos,
        "exacto": exacto.resumen(),
        "categoria": categoria.resumen(),
        "negacion": {
            "menciones_negadas": neg_total,
            "confirmadas_por_error": neg_error,
            "exactitud": round(1 - neg_error / neg_total, 4) if neg_total else None,
        },
        "evidencia_localizada": round(evid_ok / evid_total, 4) if evid_total else None,
        "ajustes_negex": ajustes,
        "latencia_media_s": round(statistics.mean(lat), 3) if lat else None,
        "latencia_mediana_s": round(statistics.median(lat), 3) if lat else None,
        "latencia_llm_media_s": round(statistics.mean(lat_llm), 3) if lat_llm else None,
        "tokens_entrada_total": tin,
        "tokens_salida_total": tout,
        "detalle": detalle,
    }
