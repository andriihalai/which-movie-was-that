from qdrant_client import QdrantClient
from sentence_transformers import CrossEncoder, SentenceTransformer

from finder import MovieFinder, QdrantMovieFinder


def ask_for_movie_title(query: str, model: MovieFinder, limit: int = 5):
    return model.find_relevant_chunks(query, limit=limit)


def main():
    client = QdrantClient("localhost", port=6333)
    embedder = SentenceTransformer("intfloat/e5-small-v2")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    finder = QdrantMovieFinder(client, embedder, reranker)

    while True:
        user_input = input("Describe the movie (or 'exit' to quit): ")
        if user_input.lower() == "exit":
            break
        chunks = ask_for_movie_title(user_input, finder)
        for chunk in chunks:
            print(f"score: {chunk.score:.3f}, title: {chunk.title}, uuid: {chunk.uuid}")


if __name__ == "__main__":
    main()
