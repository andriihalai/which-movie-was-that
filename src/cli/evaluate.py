import json
from pathlib import Path

from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from finder import QdrantMovieFinder

ROOT = Path(__file__).resolve().parents[2]
EVAL_SET_PATH = ROOT / "tests" / "eval-set.jsonl"
CHUNKED_DATA_PATH = ROOT / "src" / "ingest" / "chunked_data.jsonl"
RESULTS_PATH = ROOT / "tests" / "eval-results.json"
TOP_K = 10


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f]


def build_uuid_to_doc_id(path: Path) -> dict[str, str]:
    mapping = {}
    with path.open() as f:
        for line in f:
            d = json.loads(line)
            mapping[d["uuid"]] = d["doc_id"]
    return mapping


def rank_of_expected(candidates, expected_doc_id: str) -> int | None:
    for rank, candidate in enumerate(candidates, start=1):
        if candidate.doc_id == expected_doc_id:
            return rank
    return None


def recall_at(ranks: list[int | None], k: int) -> float:
    hits = sum(1 for rank in ranks if rank is not None and rank <= k)
    return hits / len(ranks)


def main():
    eval_set = load_jsonl(EVAL_SET_PATH)
    uuid_to_doc_id = build_uuid_to_doc_id(CHUNKED_DATA_PATH)

    client = QdrantClient("localhost", port=6333)
    embedder = SentenceTransformer("intfloat/e5-small-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    finder = QdrantMovieFinder(client, embedder, reranker)

    print(f"Running {len(eval_set)} queries (top-{TOP_K})...\n")
    print(f"{'ID':<5}{'RANK':<7}{'TITLE'}")
    print("-" * 60)

    per_query: list[dict] = []
    ranks: list[int | None] = []

    for item in eval_set:
        expected_doc_id = uuid_to_doc_id[item["relevant_chunk_uuid"]]
        candidates = finder.find(item["query"], limit=TOP_K)
        rank = rank_of_expected(candidates, expected_doc_id)
        ranks.append(rank)

        rank_str = str(rank) if rank is not None else "miss"
        print(f"{item['query_id']:<5}{rank_str:<7}{item['expected_title']}")

        per_query.append({
            "query_id": item["query_id"],
            "query": item["query"],
            "title": item["expected_title"],
            "uuid": item["relevant_chunk_uuid"],
            "rank": rank,
        })

    metrics = {
        "recall@1": round(recall_at(ranks, 1), 4),
        "recall@5": round(recall_at(ranks, 5), 4),
        "recall@10": round(recall_at(ranks, 10), 4),
    }

    print("\n" + "=" * 60)
    for name, value in metrics.items():
        print(f"{name:<12}{value}")
    print("=" * 60)

    config = {
        "embedder": "intfloat/e5-small-v2",
        "reranker": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "candidate_chunks": QdrantMovieFinder.CANDIDATE_CHUNKS,
        "top_k": TOP_K,
        "eval_set_size": len(eval_set),
    }

    with RESULTS_PATH.open("w") as f:
        json.dump(
            {"metrics": metrics, "config": config, "per_query": per_query},
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\nResults saved to {RESULTS_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
