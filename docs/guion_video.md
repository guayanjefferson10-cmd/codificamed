# Guion sugerido para el video (máx. 10 min, con tu rostro en una esquina)

Graba con OBS Studio o QuickTime + Photo Booth: captura de pantalla y cámara en una esquina.

| Min. | Qué mostrar |
|---|---|
| 0:00–0:45 | Presentación: nombre, asignatura, problema (codificación CIE-10 manual, glosas). |
| 0:45–2:00 | Arquitectura: diagrama del README; menciona las 6 fases de CRISP-ML(Q). |
| 2:00–3:00 | Código: `.env` con `LLM_PROVIDER` (sin mostrar claves), `negacion.py`, `recuperador.py`, `pipeline.py`. |
| 3:00–5:30 | App con **Ollama**: caso F01 (HTA, DM2, "niega fiebre"); muestra resaltado, estados y validación. Caso F23 (errores de tipeo). Abre el expander del pipeline (NegEx, consultas, candidatos). |
| 5:30–6:30 | Cambia a **API Claude** en la barra lateral y codifica el mismo caso. Pestaña **Comparar local vs API**. |
| 6:30–8:30 | Terminal: `python scripts/evaluar.py --provider ollama` (corta la espera) y `python scripts/comparar.py`; pestaña **Evaluación** con tabla y gráfico. |
| 8:30–9:30 | Conclusiones y limitaciones. |

Antes de grabar: ejecuta `ollama serve` (o abre la app de Ollama) y corre una nota de prueba
para que el modelo ya esté cargado en memoria y la primera respuesta no demore.
