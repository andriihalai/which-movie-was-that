# Which Movie Was That?

A RAG system that helps people recall a movie by a vague description of its plot, a scene, or a small detail.

## Scenario

The user types what they remember in their own words — for example, *"a boy stays home alone on Christmas and defends the house from burglars"* — and the system returns the most likely movie with its title, release year, and a short explanation of why it matches. The target user is a regular person, not a film expert, who only has fragments of memory and wants a quick answer without digging through forums.

## Tech Stack

- **Language:** Python
- **Vector store:** PostgreSQL with the `pgvector` extension
- **Embedding model:** `intfloat/multilingual-e5-base` (Ukrainian + English)
- **Generation model:** `Qwen3-4B` (runs locally)
- **Dataset:** [Coder-Dragon/wikipedia-movies](https://huggingface.co/datasets/Coder-Dragon/wikipedia-movies) (Apache 2.0)

## Project Structure

```
.
├── src/        # application code
├── config/     # configuration files
├── reports/    # lab reports and documentation
├── docker/     # init scripts for docker services
└── docker-compose.yml
```

## Setup

### Prerequisites

Install the tools below by following their official guides:

- [Python 3.11+](https://www.python.org/downloads/)
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)
- [Git](https://git-scm.com/downloads)

The embedding and generation models are pulled from Hugging Face on first run — see the [`sentence-transformers`](https://www.sbert.net/docs/installation.html) and [`transformers`](https://huggingface.co/docs/transformers/installation) docs if you need details on how model caching works.

PostgreSQL with the `pgvector` extension is started via Docker Compose, so no manual Postgres install is needed. See the [pgvector docs](https://github.com/pgvector/pgvector) for the extension itself.

## License

See [LICENSE](./LICENSE).
