"""Interfaz de CodificaMed (Streamlit).

    streamlit run app.py
"""
from __future__ import annotations

import html
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from codificamed.catalog import cargar_catalogo  # noqa: E402
from codificamed.config import DATA_DIR, RESULTS_DIR, get_settings  # noqa: E402
from codificamed.evaluacion import cargar_casos, evaluar  # noqa: E402
from codificamed.llm import ErrorLLM  # noqa: E402
from codificamed.pipeline import CodificaMed, Resultado  # noqa: E402
from codificamed.recuperador import Recuperador  # noqa: E402

st.set_page_config(page_title="CodificaMed · CIE-10 con PLN", page_icon="🩺", layout="wide")

ETIQ_PROV = {"ollama": "Local · Ollama", "anthropic": "API · Claude (Anthropic)",
             "openai": "API · OpenAI / compatible", "rag": "Línea base · solo RAG (sin LLM)"}
ESTADOS = {
    "confirmado": ("Confirmado", "#e3f4ea", "#0f6b3a", "✔"),
    "negado": ("Negado / descartado", "#fbe7e6", "#a12622", "✘"),
    "sospecha": ("Sospecha", "#fdf1d8", "#8a5a00", "?"),
}

st.markdown("""
<style>
.nota {font-size: 1.02rem; line-height: 1.9; padding: 1rem 1.2rem; border-radius: 10px;
       border: 1px solid rgba(128,128,128,.25); background: rgba(128,128,128,.04);}
.nota mark {padding: 1px 4px; border-radius: 4px;}
.nota sup {font-size: .68rem; font-weight: 700; margin-left: 2px;}
.tarjeta {border: 1px solid rgba(128,128,128,.25); border-radius: 10px; padding: .7rem .9rem;
          margin-bottom: .55rem;}
.tarjeta .cod {font-family: ui-monospace, monospace; font-weight: 700; font-size: 1.05rem;}
.badge {display:inline-block; font-size:.75rem; font-weight:600; padding:1px 8px;
        border-radius: 999px; margin-left: 6px;}
.evid {font-style: italic; opacity: .85;}
.aviso {font-size: .8rem; color: #8a5a00;}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ recursos
@st.cache_resource(show_spinner="Cargando catálogo CIE-10 e índice de embeddings…")
def recuperador(backend: str) -> Recuperador:
    return Recuperador(get_settings(embedding_backend=backend), cargar_catalogo())


@st.cache_data
def casos_ejemplo() -> list[dict]:
    return cargar_casos(DATA_DIR / "casos_ficticios.jsonl")


def pipeline(proveedor: str, modelo: str, negacion: str, top_k: int, backend: str) -> CodificaMed:
    campo = {"ollama": "ollama_model", "anthropic": "anthropic_model", "openai": "openai_model"}
    cambios = {"llm_provider": proveedor, "top_k_por_mencion": top_k, "embedding_backend": backend}
    if proveedor in campo and modelo:
        cambios[campo[proveedor]] = modelo
    s = get_settings(**cambios)
    return CodificaMed(s, recuperador(backend), negacion=negacion)


def estado_proveedor(p: str) -> tuple[bool, str]:
    s = get_settings()
    if p == "ollama":
        try:
            r = requests.get(s.ollama_host.rstrip("/") + "/api/tags", timeout=2)
            modelos = [m["name"] for m in r.json().get("models", [])]
            return True, f"Ollama activo · {len(modelos)} modelo(s): {', '.join(modelos[:4])}"
        except Exception:
            return False, f"Ollama no responde en {s.ollama_host} (ejecuta `ollama serve`)"
    if p == "anthropic":
        return bool(s.anthropic_api_key), "ANTHROPIC_API_KEY configurada" if s.anthropic_api_key \
            else "Falta ANTHROPIC_API_KEY en .env"
    if p == "openai":
        return bool(s.openai_api_key), "OPENAI_API_KEY configurada" if s.openai_api_key \
            else "Falta OPENAI_API_KEY en .env"
    return True, "No requiere modelo de lenguaje"


# ------------------------------------------------------------------ render
def nota_resaltada(texto: str, res: Resultado) -> str:
    spans = sorted([c for c in res.codigos if c.inicio is not None], key=lambda c: (c.inicio, -c.fin))
    partes, pos = [], 0
    agrupados: list[list] = []
    for c in spans:  # agrupa evidencias que se solapan
        if agrupados and c.inicio < agrupados[-1][1]:
            g = agrupados[-1]
            g[1] = max(g[1], c.fin)
            g[2].append(c)
        else:
            agrupados.append([c.inicio, c.fin, [c]])
    for ini, fin, cods in agrupados:
        partes.append(html.escape(texto[pos:ini]))
        estado = "confirmado" if any(c.estado == "confirmado" for c in cods) else cods[0].estado
        _, fondo, tinta, icono = ESTADOS[estado]
        etiqueta = " ".join(f"{ESTADOS[c.estado][3]}{c.codigo}" for c in cods)
        titulo = html.escape("; ".join(f"{c.codigo} {c.descripcion} ({c.estado})" for c in cods))
        partes.append(f'<mark style="background:{fondo};color:{tinta}" title="{titulo}">'
                      f'{html.escape(texto[ini:fin])}<sup>{html.escape(etiqueta)}</sup></mark>')
        pos = fin
    partes.append(html.escape(texto[pos:]))
    return f'<div class="nota">{"".join(partes)}</div>'


def tarjeta(c) -> str:
    nombre, fondo, tinta, icono = ESTADOS[c.estado]
    avisos = []
    if not c.en_candidatos:
        avisos.append("código propuesto fuera de los candidatos del RAG")
    if c.ajustado_por_negex:
        avisos.append(f"el LLM lo marcó «{c.estado_llm}», NegEx detectó {c.estado_negex}: se corrigió")
    elif c.estado_negex and c.estado_llm and c.estado_negex != c.estado_llm:
        avisos.append(f"NegEx sugiere «{c.estado_negex}»")
    if c.inicio is None:
        avisos.append("evidencia no localizada en la nota")
    aviso_html = "".join(f'<div class="aviso">⚠ {html.escape(a)}</div>' for a in avisos)
    just = f"<div>{html.escape(c.justificacion)}</div>" if c.justificacion else ""
    return (f'<div class="tarjeta"><span class="cod">{c.codigo}</span>'
            f'<span class="badge" style="background:{fondo};color:{tinta}">{icono} {nombre}</span>'
            f'<div>{html.escape(c.descripcion)}</div>'
            f'<div class="evid">Evidencia: «{html.escape(c.evidencia)}»</div>{just}{aviso_html}</div>')


def mostrar_resultado(res: Resultado, texto: str, clave: str, validar: bool = True):
    conf = [c for c in res.codigos if c.estado == "confirmado"]
    otros = [c for c in res.codigos if c.estado != "confirmado"]
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Códigos sugeridos", len(conf))
    m2.metric("Negados / sospecha", len(otros))
    m3.metric("Tiempo total", f"{res.latencia_total_s:.1f} s")
    m4.metric("Tokens (entrada/salida)", f"{res.tokens_entrada}/{res.tokens_salida}")
    izq, der = st.columns([1.15, 1])
    with izq:
        st.markdown("**Nota con evidencia resaltada**")
        st.markdown(nota_resaltada(texto, res), unsafe_allow_html=True)
        st.caption("✔ verde = confirmado · ✘ rojo = negado/descartado · ? ámbar = sospecha. "
                   "Pasa el cursor sobre una marca para ver el código.")
    with der:
        st.markdown("**Códigos CIE-10 sugeridos**")
        aceptados = []
        for c in conf + otros:
            st.markdown(tarjeta(c), unsafe_allow_html=True)
            if validar and c.estado == "confirmado":
                if st.checkbox(f"Aceptar {c.codigo}", value=True, key=f"{clave}-{c.codigo}"):
                    aceptados.append(c)
        if not res.codigos:
            st.info("El sistema no sugirió códigos para esta nota.")
        if validar and conf:
            datos = {"nota": texto, "proveedor": res.proveedor, "modelo": res.modelo,
                     "codigos_validados": [{"codigo": c.codigo, "descripcion": c.descripcion,
                                            "evidencia": c.evidencia} for c in aceptados]}
            st.download_button("⬇ Descargar codificación validada (JSON)",
                               json.dumps(datos, ensure_ascii=False, indent=2),
                               file_name="codificacion_validada.json", mime="application/json",
                               key=f"dl-{clave}")
    for a in res.advertencias:
        st.caption(f"⚠ {a}")


def detalle_pipeline(res: Resultado):
    nota = res.nota
    with st.expander("🔎 Pipeline de PLN: preprocesamiento y negación (NegEx)"):
        st.markdown("**Oraciones con alcance de negación** (`[NEG …]`, `[SOSP …]`)")
        for i, o in enumerate(nota.marcado, 1):
            st.code(f"{i}. {o}", language=None)
        filas = [{"cláusula (original)": c.texto, "estado NegEx": c.estado,
                  "consulta normalizada (RAG)": c.consulta,
                  "correcciones": ", ".join(f"{a}→{b}" for a, b in c.correcciones)}
                 for c in nota.clausulas]
        st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)
    with st.expander(f"📚 Candidatos recuperados del catálogo CIE-10 ({len(res.candidatos)})"):
        st.dataframe(pd.DataFrame([{"código": c.codigo, "descripción": c.descripcion,
                                    "similitud": round(c.score, 3), "consulta": c.consulta}
                                   for c in res.candidatos]),
                     width="stretch", hide_index=True)
    if res.respuesta_cruda:
        with st.expander("🧾 Respuesta cruda del modelo"):
            st.code(res.respuesta_cruda, language="json")


def ejecutar(prov, modelo, negacion, top_k, backend, texto) -> Resultado | None:
    try:
        return pipeline(prov, modelo, negacion, top_k, backend).codificar(texto)
    except ErrorLLM as e:
        st.error(str(e))
    except Exception as e:  # noqa: BLE001
        st.error(f"Error inesperado: {e}")
    return None


# ------------------------------------------------------------------ barra lateral
s0 = get_settings()
with st.sidebar:
    st.title("🩺 CodificaMed")
    st.caption("Codificación automática CIE-10 de notas clínicas con PLN")
    proveedores = list(ETIQ_PROV)
    proveedor = st.selectbox("Modelo de lenguaje", proveedores,
                             index=proveedores.index(s0.llm_provider) if s0.llm_provider in proveedores else 0,
                             format_func=ETIQ_PROV.get)
    modelo_def = {"ollama": s0.ollama_model, "anthropic": s0.anthropic_model,
                  "openai": s0.openai_model}.get(proveedor, "")
    modelo = st.text_input("Modelo", modelo_def, disabled=proveedor == "rag")
    ok, msg = estado_proveedor(proveedor)
    (st.success if ok else st.warning)(msg)
    with st.expander("Opciones avanzadas"):
        backends = ["hibrido", "e5", "tfidf"]
        backend = st.selectbox("Búsqueda del RAG", backends,
                               index=backends.index(s0.embedding_backend) if s0.embedding_backend in backends else 0,
                               help="hibrido = e5 + TF-IDF; e5 = multilingual-e5-small (semántico); "
                                    "tfidf = n-gramas de caracteres")
        top_k = st.slider("Candidatos por mención (top-k)", 3, 15, s0.top_k_por_mencion)
        negacion = st.radio("Política de negación", ["hibrida", "llm"],
                            format_func={"hibrida": "Híbrida (NegEx corrige al LLM)",
                                         "llm": "Solo el LLM decide"}.get)
    st.divider()
    st.caption("Datos ficticios. Prohibido usar historias clínicas reales. "
               "Las sugerencias deben ser validadas por un codificador humano.")

tab_cod, tab_cmp, tab_eval, tab_info = st.tabs(
    ["🩺 Codificar nota", "⚖️ Comparar local vs API", "📊 Evaluación", "ℹ️ Acerca de"])

# ------------------------------------------------------------------ pestaña 1
with tab_cod:
    casos = casos_ejemplo()
    opciones = ["(escribir nota propia)"] + [f"{c['id']} · {c['texto'][:70]}…" for c in casos]
    sel = st.selectbox("Cargar un caso clínico ficticio de ejemplo", opciones, index=1)
    texto_def = casos[opciones.index(sel) - 1]["texto"] if sel != opciones[0] else ""
    texto = st.text_area("Nota clínica (texto libre)", texto_def, height=150, key=f"nota-{sel}")
    if sel != opciones[0]:
        caso = casos[opciones.index(sel) - 1]
        st.caption(f"Códigos de referencia del caso: {', '.join(caso['codigos'])}"
                   + (f" · negados: {', '.join(caso['negados'])}" if caso["negados"] else ""))
    if st.button("Sugerir códigos CIE-10", type="primary", disabled=not texto.strip()):
        with st.spinner(f"Codificando con {ETIQ_PROV[proveedor]}…"):
            st.session_state["res"] = (ejecutar(proveedor, modelo, negacion, top_k, backend, texto), texto)
    if st.session_state.get("res") and st.session_state["res"][0]:
        res, txt = st.session_state["res"]
        st.markdown(f"##### Resultado · {ETIQ_PROV.get(res.proveedor, res.proveedor)} · `{res.modelo}`")
        mostrar_resultado(res, txt, "cod")
        detalle_pipeline(res)

# ------------------------------------------------------------------ pestaña 2
with tab_cmp:
    st.markdown("Ejecuta la misma nota con dos configuraciones y compara qué códigos coinciden.")
    c1, c2 = st.columns(2)
    pa = c1.selectbox("Configuración A", proveedores, index=0, format_func=ETIQ_PROV.get, key="pa")
    pb = c2.selectbox("Configuración B", proveedores, index=1, format_func=ETIQ_PROV.get, key="pb")
    texto_cmp = st.text_area("Nota clínica", st.session_state.get(f"nota-{sel}", texto_def),
                             height=120, key="nota-cmp")
    if st.button("Comparar", type="primary", disabled=not texto_cmp.strip()):
        salida = {}
        for etiqueta, p in (("A", pa), ("B", pb)):
            m = {"ollama": s0.ollama_model, "anthropic": s0.anthropic_model,
                 "openai": s0.openai_model}.get(p, "")
            with st.spinner(f"{ETIQ_PROV[p]}…"):
                salida[etiqueta] = ejecutar(p, m, negacion, top_k, backend, texto_cmp)
        st.session_state["cmp"] = (salida, texto_cmp)
    if st.session_state.get("cmp"):
        salida, txt = st.session_state["cmp"]
        ra, rb = salida.get("A"), salida.get("B")
        if ra and rb:
            ca = {c.codigo: c for c in ra.confirmados}
            cb = {c.codigo: c for c in rb.confirmados}
            todos = sorted(set(ca) | set(cb))
            filas = [{"código": k, "descripción": (ca.get(k) or cb.get(k)).descripcion,
                      f"A · {ra.proveedor}": "✔" if k in ca else "—",
                      f"B · {rb.proveedor}": "✔" if k in cb else "—"} for k in todos]
            acuerdo = len(set(ca) & set(cb)) / len(todos) if todos else 1.0
            k1, k2, k3 = st.columns(3)
            k1.metric("Acuerdo (Jaccard)", f"{acuerdo:.0%}")
            k2.metric(f"Tiempo A · {ra.proveedor}", f"{ra.latencia_total_s:.1f} s")
            k3.metric(f"Tiempo B · {rb.proveedor}", f"{rb.latencia_total_s:.1f} s")
            st.dataframe(pd.DataFrame(filas), width="stretch", hide_index=True)
        cols = st.columns(2)
        for col, (etq, r) in zip(cols, salida.items()):
            with col:
                if r:
                    st.markdown(f"**{etq} · {ETIQ_PROV.get(r.proveedor)} · `{r.modelo}`**")
                    st.markdown(nota_resaltada(txt, r), unsafe_allow_html=True)
                    for c in r.codigos:
                        st.markdown(tarjeta(c), unsafe_allow_html=True)

# ------------------------------------------------------------------ pestaña 3
with tab_eval:
    st.markdown("Métricas sobre casos con códigos conocidos. Las corridas completas se lanzan con "
                "`python scripts/evaluar.py --provider …`; aquí puedes verlas o lanzar una rápida.")
    metricas = []
    for p in sorted(RESULTS_DIR.glob("*/metricas.json")):
        m = json.loads(p.read_text(encoding="utf-8"))
        m["carpeta"] = p.parent.name
        metricas.append(m)
    if metricas:
        tabla = pd.DataFrame([{
            "dataset": m["dataset"], "proveedor": ETIQ_PROV.get(m["proveedor"], m["proveedor"]),
            "modelo": m["modelo"], "negación": m.get("politica_negacion", ""), "casos": m["casos"],
            "P": m["exacto"]["precision"], "R": m["exacto"]["recall"], "F1": m["exacto"]["f1"],
            "F1 categoría": m["categoria"]["f1"],
            "negados confirmados": f"{m['negacion']['confirmadas_por_error']}/{m['negacion']['menciones_negadas']}",
            "latencia media (s)": m["latencia_media_s"],
        } for m in metricas])
        st.dataframe(tabla, width="stretch", hide_index=True)
        for img in ("comparacion.png", "comparacion_categoria.png"):
            if (RESULTS_DIR / img).exists():
                st.image(str(RESULTS_DIR / img), width=760)
        elegida = st.selectbox("Ver detalle por caso de la corrida", [m["carpeta"] for m in metricas])
        m = next(x for x in metricas if x["carpeta"] == elegida)
        st.dataframe(pd.DataFrame([{
            "caso": d["id"], "esperado": ", ".join(d["gold"]), "predicho": ", ".join(d["pred"]),
            "aciertos": ", ".join(d["exacto"]["tp"]), "falsos +": ", ".join(d["exacto"]["fp"]),
            "omitidos": ", ".join(d["exacto"]["fn"]), "negados confirmados": ", ".join(d["negados_confirmados"]),
        } for d in m["detalle"]]), width="stretch", hide_index=True)
    else:
        st.info("Aún no hay resultados guardados en results/.")

    with st.expander("▶ Evaluación rápida desde la interfaz"):
        n = st.slider("Número de casos ficticios", 3, 40, 10)
        if st.button(f"Evaluar con {ETIQ_PROV[proveedor]}"):
            casos_ev = casos_ejemplo()[:n]
            pipe = pipeline(proveedor, modelo, negacion, top_k, backend)
            barra = st.progress(0.0)
            preds = []
            for i, caso in enumerate(casos_ev, 1):
                try:
                    preds.append(pipe.codificar(caso["texto"]).a_dict())
                except Exception as e:  # noqa: BLE001
                    preds.append({"error": str(e)})
                barra.progress(i / n, text=f"Caso {caso['id']} ({i}/{n})")
            r = evaluar(casos_ev, preds)
            a, b, c, d = st.columns(4)
            a.metric("Precisión", f"{r['exacto']['precision']:.2f}")
            b.metric("Recall", f"{r['exacto']['recall']:.2f}")
            c.metric("F1", f"{r['exacto']['f1']:.2f}")
            d.metric("F1 categoría", f"{r['categoria']['f1']:.2f}")

# ------------------------------------------------------------------ pestaña 4
with tab_info:
    st.markdown("""
### ¿Qué hace CodificaMed?
Recibe una nota clínica en texto libre y sugiere códigos **CIE-10** con la **frase que justifica**
cada uno, distinguiendo diagnósticos **confirmados**, **negados/descartados** y de **sospecha**.
El codificador humano solo valida.

### Pipeline de PLN
1. **Preprocesamiento**: limpieza, segmentación en oraciones y cláusulas, expansión de
   abreviaturas (HTA, DM2, EDA…), corrección ortográfica contra el vocabulario del catálogo.
2. **Negación (NegEx-es)**: reglas de disparadores pre/post y alcance ("niega", "se descarta",
   "sin", "negativo", "probable"…).
3. **Representación y recuperación (RAG)**: embeddings `multilingual-e5-small` del catálogo
   CIE-10 (OMS, español) enriquecido con sinónimos; búsqueda por similitud coseno por cláusula.
4. **Modelado (LLM)**: el modelo (local con Ollama o API externa) elige los códigos entre los
   candidatos y devuelve evidencia literal y estado en JSON.
5. **Validación**: código en el catálogo, evidencia ubicada en la nota, coherencia con NegEx.

El proveedor se cambia en `.env` (`LLM_PROVIDER=ollama|anthropic|openai|rag`) o en esta barra lateral.
""")
