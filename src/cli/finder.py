from dataclasses import dataclass
from typing import Protocol

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer


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

        candidates = sorted(best_per_movie.values(), key=lambda c: c.score, reverse=True)
        return candidates[:limit]
