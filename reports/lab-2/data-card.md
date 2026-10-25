# Data Card — Which Movie Was That?

## Source and License

- **Dataset:** [Coder-Dragon/wikipedia-movies](https://huggingface.co/datasets/Coder-Dragon/wikipedia-movies)
- **Source:** Hugging Face Hub, scraped from English Wikipedia movie articles.
- **License:** Apache License 2.0
- **Access:** Public, downloaded via the `datasets` library.

## Corpus Size

| Stage           | Documents | Notes                                                                    |
| --------------- | --------- | ------------------------------------------------------------------------ |
| Raw dataset     | 30671     | Number of rows in the dataset as downloaded.                             |
| After filtering | 10951     | Only American films released after 1950 were kept to reduce corpus size. |
| After cleaning  | 10918     | Rows with an empty `Plot` or a `Plot` shorter than 100 characters were dropped. |

### Cleaning Steps Applied

- Unicode normalization (NFC) and encoding fixes via `ftfy`.
- HTML stripping (`BeautifulSoup`) and HTML-entity unescaping.
- Removal of Wikipedia citation markers (`[1]`, `[2]`, ...).
- Collapsing of repeated whitespace and blank lines.
- Dropping rows with empty plot after cleaning.

## Chunking

Fixed-size word-based chunks with 15% overlap, produced from the cleaned `Plot` field.

| Parameter                  | Value            |
| -------------------------- | ---------------- |
| Chunk size (words)         | 350              |
| Overlap ratio              | 15% (≈ 52 words) |
| Total chunks               | 21892            |
| Average chunk size (words) | 264              |
| Median chunk size (words)  | 350              |

Each chunk is stored with:

- `uuid` — globally stable chunk identifier,
- `doc_id` — source document identifier (Wikipedia page URL),
- `chunk_index_in_document` — sequential position of the chunk within the source document,
- `text` — the chunk text.

## Top-20 Most Frequent Words

Computed on the cleaned, chunked corpus after lowercasing and removing English stopwords.

| Rank | Word   | Count  |
| ---- | ------ | ------ |
| 1    | one    | 12,433 |
| 2    | back   | 10,727 |
| 3    | two    | 9,403  |
| 4    | tells  | 9,140  |
| 5    | new    | 8,637  |
| 6    | home   | 8,144  |
| 7    | later  | 7,674  |
| 8    | house  | 7,272  |
| 9    | time   | 7,259  |
| 10   | father | 7,074  |
| 11   | find   | 7,041  |
| 12   | man    | 6,872  |
| 13   | finds  | 6,808  |
| 14   | get    | 6,511  |
| 15   | also   | 6,455  |
| 16   | police | 6,086  |
| 17   | life   | 6,071  |
| 18   | go     | 5,991  |
| 19   | takes  | 5,837  |
| 20   | goes   | 5,781  |

## Known Limitations

- **English only.** All plot descriptions are in English. Non-English queries are supported only through the multilingual embedding model, which bridges the language gap at retrieval time but may lose nuance.
- **Wikipedia bias.** Only films with a Wikipedia article are included. Lesser-known films, short films, and non-English-language cinema are underrepresented.
- **American film subset.** After filtering, the corpus is skewed toward American productions, which further narrows cultural and genre coverage.
- **Plot summaries, not full scripts.** Chunks come from editor-written synopses, so specific lines of dialogue, minor characters, or background details are often absent.
- **Spoiler-heavy text.** Wikipedia plot sections typically describe the full story including the ending. Queries about late-film twists may match more easily than queries about opening scenes.
- **No cast / crew metadata used.** Only the plot text is embedded. Queries like "the movie with actor X" cannot be answered from embeddings alone.
- **Static snapshot.** The dataset is a one-time download; new or recently updated Wikipedia content is not reflected until re-ingestion.
- **Potential duplicates.** Remakes, re-releases, and alternate-cut articles can produce near-duplicate chunks that may dominate retrieval results.
- **Noise from Wikipedia markup.** Despite cleaning, some residual artifacts (lists, section headers, template remnants) may remain in long articles.
