"""Count the top-20 most frequent words in chunked_data.jsonl."""
import json
import re
from collections import Counter
from pathlib import Path

import nltk
from nltk.corpus import stopwords

try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords")
    STOPWORDS = set(stopwords.words("english"))

INPUT_PATH = Path(__file__).parent / "chunked_data.jsonl"
TOP_N = 20
TOKEN_RE = re.compile(r"[a-z']+")


def tokens(text: str):
    for word in TOKEN_RE.findall(text.lower()):
        word = word.strip("'")
        if len(word) > 1 and word not in STOPWORDS:
            yield word


def main() -> None:
    counter: Counter[str] = Counter()
    chunk_count = 0

    with open(INPUT_PATH, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            counter.update(tokens(record["text"]))
            chunk_count += 1

    print(f"Processed {chunk_count} chunks, {sum(counter.values())} tokens, "
          f"{len(counter)} unique words.\n")
    print(f"Top-{TOP_N} words:")
    print(f"{'Rank':<6}{'Word':<20}{'Count':>10}")
    print("-" * 36)
    for rank, (word, count) in enumerate(counter.most_common(TOP_N), start=1):
        print(f"{rank:<6}{word:<20}{count:>10,}")


if __name__ == "__main__":
    main()
