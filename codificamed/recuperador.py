"""Recuperador semántico sobre el catálogo CIE-10 (la "R" de RAG).

Backends:
  - hibrido (recomendado): combina e5 y TF-IDF (0.5 / 0.5). Lo semántico encuentra sinónimos y
           lo léxico premia la coincidencia literal con los sinónimos del catálogo (HTA -> I10).
  - e5:    embeddings densos multilingües (intfloat/multilingual-e5-small, 384 dim) con
           sentence-transformers. Captura sinónimos ("HTA" ~ "hipertensión esencial").
  - tfidf: TF-IDF de n-gramas de caracteres (3-5). No necesita descargar modelos y es
           robusto a errores de tipeo; sirve como respaldo y como comparación.

El índice (matriz normalizada de los ~12 000 códigos) se calcula una vez y se guarda en
.cache/ para que las siguientes ejecuciones arranquen en segundos.
"""
from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass

import numpy as np

from .catalog import Catalogo, cargar_catalogo
from .config import CACHE_DIR, Settings, get_settings


@dataclass
class Candidato:
    codigo: str
    descripcion: str
    score: float
    consulta: str = ""


class Recuperador:
    def __init__(self, settings: Settings | None = None, catalogo: Catalogo | None = None):
        self.s = settings or get_settings()
        self.catalogo = catalogo or cargar_catalogo()
        self.backend = self.s.embedding_backend
        self._modelo = None
        self._vectorizer = None
        if self.backend == "hibrido":
            self._partes = [Recuperador(self.s.con(embedding_backend=b), self.catalogo)
                            for b in ("e5", "tfidf")]
            self.matriz = None
        else:
            self._partes = []
            self.matriz = self._cargar_o_construir()

    # ------------------------------------------------------------------ índice
    def _clave_cache(self) -> str:
        h = hashlib.md5("\n".join(self.catalogo.textos).encode()).hexdigest()[:10]
        nombre = self.s.embedding_model.replace("/", "_") if self.backend == "e5" else "char35"
        return f"indice_{self.backend}_{nombre}_{h}"

    def _cargar_o_construir(self):
        CACHE_DIR.mkdir(exist_ok=True)
        ruta = CACHE_DIR / f"{self._clave_cache()}.pkl"
        if ruta.exists():
            with open(ruta, "rb") as f:
                datos = pickle.load(f)
            self._vectorizer = datos.get("vectorizer")
            return datos["matriz"]
        textos = self.catalogo.textos
        if self.backend == "e5":
            m = self._encoder().encode([f"passage: {t}" for t in textos], batch_size=128,
                                       normalize_embeddings=True, show_progress_bar=True)
            m = np.asarray(m, dtype=np.float32)
            datos = {"matriz": m}
        elif self.backend == "tfidf":
            from sklearn.feature_extraction.text import TfidfVectorizer

            self._vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                               min_df=2, sublinear_tf=True, strip_accents="unicode",
                                               lowercase=True)
            m = self._vectorizer.fit_transform(textos)
            datos = {"matriz": m, "vectorizer": self._vectorizer}
        else:
            raise ValueError(f"EMBEDDING_BACKEND desconocido: {self.backend} (use e5 o tfidf)")
        with open(ruta, "wb") as f:
            pickle.dump(datos, f)
        return datos["matriz"]

    def _encoder(self):
        if self._modelo is None:
            from sentence_transformers import SentenceTransformer

            self._modelo = SentenceTransformer(self.s.embedding_model)
        return self._modelo

    # ------------------------------------------------------------------ búsqueda
    def _vectorizar(self, consultas: list[str]):
        if self.backend == "e5":
            return np.asarray(self._encoder().encode([f"query: {q}" for q in consultas],
                                                     normalize_embeddings=True), dtype=np.float32)
        return self._vectorizer.transform(consultas)

    def similitudes(self, consultas: list[str]) -> np.ndarray:
        """Matriz (consultas x códigos) de similitud coseno."""
        if self._partes:
            densa, lexica = (p.similitudes(consultas) for p in self._partes)
            return 0.5 * densa + 0.5 * lexica
        sims = self._vectorizar(consultas) @ self.matriz.T
        if hasattr(sims, "toarray"):
            sims = sims.toarray()
        return np.asarray(sims)

    def buscar(self, consultas: list[str], k: int = 8) -> list[list[Candidato]]:
        if not consultas:
            return []
        sims = self.similitudes(consultas)
        resultados = []
        codigos = self.catalogo.codigos
        for fila, consulta in zip(sims, consultas):
            top = np.argpartition(-fila, min(k, len(fila) - 1))[:k]
            top = top[np.argsort(-fila[top])]
            resultados.append([
                Candidato(codigos[i], self.catalogo.descripcion(codigos[i]), float(fila[i]), consulta)
                for i in top
            ])
        return resultados

    @property
    def umbral_por_defecto(self) -> float:
        return {"e5": 0.86, "tfidf": 0.45, "hibrido": 0.66}.get(self.backend, 0.5)
