# Comparación de proveedores

| Dataset | Proveedor | Modelo | Negación | P (exacto) | R (exacto) | F1 (exacto) | F1 (categoría) | Negados confirmados | Latencia media (s) | Tokens entrada/salida |
|---|---|---|---|---|---|---|---|---|---|---|
| casos_ficticios.jsonl | Línea base (solo RAG) | rag-hibrido | n/a | 0.359 | 0.569 | **0.440** | 0.581 | 0/40 | 0.17 | 0/0 |
| casos_ficticios.jsonl | Local (Ollama) | qwen2.5:7b | hibrida | 0.394 | 0.707 | **0.506** | 0.625 | 0/40 | 21.393 | 75946/10042 |
| casos_ficticios.jsonl | Local (Ollama) | qwen2.5:7b | llm | 0.394 | 0.707 | **0.506** | 0.625 | 0/40 | 21.358 | 75946/10042 |
| casos_ficticios.jsonl | API (OpenAI) | openai/gpt-oss-120b | hibrida | 0.915 | 0.931 | **0.923** | 0.974 | 0/40 | 16.046 | 69831/29195 |
