# CodificaMed 🩺

**Codificación automática CIE-10 de notas clínicas con Procesamiento de Lenguaje Natural.**

CodificaMed recibe una nota clínica en texto libre, como las de la Red Pública Integral de Salud,
y sugiere sus códigos **CIE-10**. Para cada código muestra la **frase que lo justifica** y si el
diagnóstico está **confirmado**, **negado/descartado** o en **sospecha**. Así el codificador
humano valida la propuesta en lugar de codificar desde cero.

> Asignatura **Base de Conocimiento** — ESPOCH · FIE · Escuela de Ingeniería de Software.
> Unidad: Procesamiento de Lenguaje Natural. Docente: Ing. Isaac Torres.
> Solo se usan **casos clínicos ficticios** y el corpus público **CodiEsp**. No se usan historias clínicas reales.

---

## Arquitectura

```
 Nota clínica (texto libre)
        │
        ▼
 ┌─────────────────────────────┐
 │ 1. Preprocesamiento          │  limpieza · segmentación en oraciones y cláusulas ·
 │                              │  expansión de abreviaturas (HTA, DM2, EDA…) ·
 │                              │  corrección ortográfica contra el vocabulario CIE-10
 └──────────────┬──────────────┘
                ▼
 ┌─────────────────────────────┐
 │ 2. NegEx-es                  │  "niega", "se descarta", "sin", "negativo" → negado
 │                              │  "probable", "a descartar", "sospecha de" → sospecha
 └──────────────┬──────────────┘
                ▼
 ┌─────────────────────────────┐      ┌──────────────────────────────────┐
 │ 3. Recuperación (RAG)        │◄────►│ Índice vectorial del catálogo    │
 │    embeddings por cláusula   │      │ CIE-10 OMS en español (10 553 c.)│
 │    top-k por similitud coseno│      │ + sinónimos clínicos locales     │
 └──────────────┬──────────────┘      └──────────────────────────────────┘
                ▼
 ┌─────────────────────────────┐   LLM_PROVIDER en .env
 │ 4. LLM                       │   ├─ ollama     (local: qwen2.5, llama3.1…)
 │    elige códigos + evidencia │   ├─ anthropic  (API Claude)
 │    + estado, en JSON         │   ├─ openai     (API OpenAI / compatible)
 └──────────────┬──────────────┘   └─ rag        (línea base sin LLM)
                ▼
 ┌─────────────────────────────┐
 │ 5. Validación                │  el código existe en el catálogo · la evidencia se ubica
 │                              │  en la nota · NegEx corrige confirmaciones de negados
 └──────────────┬──────────────┘
                ▼
   Interfaz Streamlit: nota resaltada + códigos sugeridos + validación humana
```

## Requisitos

- macOS, Linux o Windows con **Python 3.10 o superior**
- [**Ollama**](https://ollama.com/download) 0.5 o superior, para el modelo local
- Opcional: una clave de API de Anthropic (Claude) u OpenAI, para comparar con un modelo externo
- Unos 6 GB libres: el modelo local ocupa cerca de 4.7 GB y el de embeddings unos 470 MB

## Instalación paso a paso (macOS)

```bash
# 1. Entrar a la carpeta del proyecto y crear un entorno virtual
cd codificamed
python3 -m venv .venv
source .venv/bin/activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Crear tu archivo de configuración
cp .env.example .env
#    Abre .env y, si vas a usar la API, pega tu ANTHROPIC_API_KEY (u OPENAI_API_KEY)

# 4. Descargar el modelo local con Ollama (con la app de Ollama abierta)
ollama pull qwen2.5:7b          # recomendado (16 GB de RAM)
# ollama pull qwen2.5:3b        # alternativa si tu Mac tiene 8 GB de RAM
#    Si usas otro modelo, cámbialo en OLLAMA_MODEL dentro de .env

# 5. Construir el índice de embeddings del catálogo (solo la primera vez, 1-3 min)
python scripts/construir_indice.py

# 6. Abrir la aplicación
streamlit run app.py
```

La aplicación se abre en `http://localhost:8501`. En la barra lateral se elige el modelo:
**Local · Ollama**, **API · Claude**, **API · OpenAI** o **Línea base · solo RAG**.

## Cambiar entre modelo local y API externa

El cambio se hace solo por configuración. En `.env`:

```ini
LLM_PROVIDER=ollama       # modelo local
LLM_PROVIDER=anthropic    # API de Claude
LLM_PROVIDER=openai       # API de OpenAI (o compatible con OPENAI_BASE_URL)
LLM_PROVIDER=rag          # sin LLM (línea base)
```

También se puede cambiar en la barra lateral de la interfaz o con `--provider` en los scripts.

## Evaluación (precisión, recall, F1)

```bash
python scripts/evaluar.py --provider rag            # línea base sin LLM
python scripts/evaluar.py --provider ollama         # modelo local
python scripts/evaluar.py --provider anthropic      # API externa
python scripts/evaluar.py --provider ollama --negacion llm   # ablación: sin corrección NegEx

python scripts/comparar.py                          # tabla y gráfico comparativos
```

Cada corrida guarda `results/<nombre>/predicciones.jsonl` y `results/<nombre>/metricas.json`.
`comparar.py` genera `results/comparacion.md`, una tabla lista para pegar en el informe, y
`results/comparacion.png`.

**Métricas reportadas**

| Métrica | Descripción |
|---|---|
| Precisión, recall y F1 (micro) | sobre el código exacto (`E11.9`) y sobre la categoría de 3 caracteres (`E11`) |
| F1 macro | promedio del F1 de cada caso |
| Negados confirmados | cuántas menciones negadas/descartadas el sistema confirmó por error |
| Evidencia localizada | % de códigos cuya evidencia se encontró en la nota |
| Latencia y tokens | tiempo por nota y consumo de tokens, para comparar costo local vs API |

### Conjuntos de datos

- **`data/casos_ficticios.jsonl`**: 40 casos clínicos ficticios redactados para el proyecto.
  Incluyen abreviaturas, errores de tipeo, negaciones y diagnósticos descartados. Cada caso trae
  los códigos esperados (`"J44.0|J44.1"` acepta cualquiera de los dos), códigos opcionales y
  los códigos negados que **no** deben confirmarse.
- **CodiEsp** (opcional): casos clínicos publicados en español y anotados con CIE-10
  (CLEF eHealth 2020). Para usarlo:
  1. Descarga `codiesp.zip` desde <https://zenodo.org/records/3837305> y descomprímelo en `data/codiesp/`.
  2. `python scripts/preparar_codiesp.py --dir data/codiesp --split test --n 100`
  3. `python scripts/evaluar.py --provider ollama --dataset data/codiesp_test.jsonl`

  CodiEsp usa CIE-10-ES, una versión más detallada. Sus códigos se reducen a la CIE-10 de la OMS
  (4 caracteres), por eso en CodiEsp la métrica principal es la de **categoría**.

## Pruebas

```bash
pytest -q
```

Las pruebas cubren la negación, las abreviaturas, la segmentación, las métricas y el pipeline
completo. Para el pipeline usan un servidor Ollama simulado, así que no necesitan modelos descargados.

## Estructura del proyecto

```
codificamed/
├── app.py                      # interfaz Streamlit
├── .env.example                # configuración de ejemplo (sin claves)
├── requirements.txt
├── codificamed/
│   ├── config.py               # lectura de .env (proveedor, modelos, parámetros)
│   ├── catalog.py              # catálogo CIE-10 + sinónimos
│   ├── abreviaturas.py         # diccionario de siglas clínicas
│   ├── negacion.py             # NegEx adaptado al español
│   ├── preprocesamiento.py     # limpieza, segmentación, normalización
│   ├── recuperador.py          # embeddings e5 / TF-IDF + búsqueda (RAG)
│   ├── prompts.py              # prompt del sistema y esquema JSON
│   ├── pipeline.py             # orquestación y validación
│   ├── evaluacion.py           # precisión, recall, F1
│   └── llm/                    # clientes Ollama, Anthropic, OpenAI
├── data/
│   ├── cie10_es.csv            # catálogo CIE-10 OMS en español (generado)
│   ├── cie10_minsal_2008.xls   # fuente original del catálogo
│   ├── sinonimos_cie10.csv     # base de conocimiento de sinónimos clínicos
│   └── casos_ficticios.jsonl   # conjunto de evaluación
├── scripts/
│   ├── build_catalog.py        # regenera cie10_es.csv desde el .xls
│   ├── construir_indice.py     # precalcula embeddings
│   ├── evaluar.py              # corre la evaluación
│   ├── comparar.py             # compara proveedores
│   └── preparar_codiesp.py     # convierte CodiEsp a JSONL
├── tests/
└── results/                    # métricas y predicciones de cada corrida
```

## Fuentes

- Catálogo CIE-10 (OMS, 2008) en español publicado por MINSAL/DEIS (Chile), redistribuido en
  <https://github.com/tamycova/icd-cie-codes-chile>.
- Miranda-Escalada, A., Gonzalez-Agirre, A., Armengol-Estapé, J. y Krallinger, M. (2020).
  *Overview of automatic clinical coding: annotations, guidelines, and solutions for non-English
  clinical cases at CodiEsp track of CLEF eHealth 2020*.
- Chapman, W. W. et al. (2001). *A simple algorithm for identifying negated findings and diseases
  in discharge summaries* (NegEx). Journal of Biomedical Informatics.
- Wang, L. et al. (2024). *Multilingual E5 Text Embeddings: A Technical Report*.

## Limitaciones

Es un prototipo académico de **apoyo** a la codificación y no reemplaza a un codificador
certificado. Ver el informe ejecutivo para el detalle de las limitaciones.

## Entregables

- **Informe ejecutivo:** [docs/informe_ejecutivo.pdf](docs/informe_ejecutivo.pdf)
- **Video demostrativo (5:40):** [ver en Microsoft Stream](https://liveespochedu-my.sharepoint.com/:v:/g/personal/bryan_guayan_espoch_edu_ec/IQC5d006F4K3SIjh6vLxrna2Ac5LdkIWLzzZnijp4lWOA_Q?e=GqF1xy)
- **Resultados de la evaluación:** [results/comparacion.md](results/comparacion.md)
