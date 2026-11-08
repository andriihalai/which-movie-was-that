from dataclasses import dataclass
from typing import Protocol
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client import models
from sentence_transformers import CrossEncoder, SentenceTransformer
from qdrant_client.models import SparseVector


@dataclass
class MovieCandidate:
    title: str
    doc_id: str
    score: float


@dataclass
class MovieChunk:
    title: str
    uuid: str
    score: float


class MovieFinder(Protocol):
    def find(self, query: str, limit: int = 5) -> list[MovieCandidate]:
        ...

    def find_relevant_chunks(self, query: str, limit: int = 5) -> list[MovieChunk]:
        ...


class QdrantMovieFinder:
    CANDIDATE_CHUNKS = 100

    def __init__(
        self,
        client: QdrantClient,
        embedder: SentenceTransformer,
        reranker: CrossEncoder,
        collection: str = "movies",
    ):
        self._client = client
        self._embedder = embedder
        self._reranker = reranker
        self._collection = collection

    def encode_query_dense(self, query: str) -> list[float]:
        vector = self._embedder.encode(
            f"query: {query}", normalize_embeddings=True
        ).tolist()
        return vector

    def encode_query_sparse(self, query: str) -> list[float]:
        bm25 = SparseTextEmbedding("Qdrant/bm25")
        q = next(bm25.query_embed(query))
        qsparse = SparseVector(indices=q.indices.tolist(),
                               values=q.values.tolist())
        return qsparse

    def find_rrf(self, query: str, limit: int = 5, candidates: int = 50, rrf_k: int = 61) -> list[MovieCandidate]:
        vector = self.encode_query_dense(query)
        qsparse = self.encode_query_sparse(query)

        result = self._client.query_points(
            collection_name="movies",
            prefetch=[
                models.Prefetch(vector,  using="dense", limit=candidates),
                models.Prefetch(query=qsparse, using="bm25",
                                limit=candidates),
            ],
            query=models.RrfQuery(rrf=models.Rrf(k=rrf_k)),   # Qdrant ≥ 1.16
            limit=limit,
            with_payload=True,
        ).points

        return result

    def find_relevant_chunks(self, query: str, limit: int = 5) -> list[MovieChunk]:
        vector = self._embedder.encode(
            f"query: {query}", normalize_embeddings=True
        ).tolist()
        result = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=limit,
            with_payload=True,
        )

        return [
            MovieChunk(point.payload["title"], str(point.id), point.score)
            for point in result.points
        ]

    def find(self, query: str, limit: int = 5) -> list[MovieCandidate]:
        vector = self._embedder.encode(
            f"query: {query}", normalize_embeddings=True
        ).tolist()
        result = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            limit=self.CANDIDATE_CHUNKS,
            with_payload=True,
        )

        pairs = [(query, point.payload["text"]) for point in result.points]
        rerank_scores = self._reranker.predict(pairs)

        best_per_movie: dict[str, MovieCandidate] = {}
        for point, score in zip(result.points, rerank_scores):
            doc_id = point.payload["doc_id"]
            score = float(score)
            existing = best_per_movie.get(doc_id)
            if existing is None or score > existing.score:
                best_per_movie[doc_id] = MovieCandidate(
                    title=point.payload["title"],
                    doc_id=doc_id,
                    score=score,
                )

        candidates = sorted(best_per_movie.values(),
                            key=lambda c: c.score, reverse=True)
        return candidates[:limit]
