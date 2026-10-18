import unicodedata
import ftfy
import html
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
import re


def to_unicode(value) -> str:
    if value is None or (not isinstance(value, (str, bytes)) and pd.isna(value)):
        return ""
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("cp1252", errors="replace")
    return str(value)


def normalize_encoding(text: str) -> str:
    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\u00a0", " ")
    # прибрати невидимі керувальні символи, але залишити \n і \t
    return "".join(ch for ch in text
                   if ch in "\n\t" or unicodedata.category(ch)[0] != "C")


warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)


def strip_html(text: str) -> str:
    if "<" in text and ">" in text:                
        soup = BeautifulSoup(text, "html.parser")
        for tag in soup(["script", "style"]):      
            tag.decompose()
        text = soup.get_text(separator=" ")
    return html.unescape(text) 


CITATION_RE = re.compile(r"\[\d+\]")
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTINEWLINE_RE = re.compile(r"\n{3,}")


def clean_wiki_noise(text: str) -> str:
    text = CITATION_RE.sub("", text)
    text = MULTISPACE_RE.sub(" ", text)
    text = MULTINEWLINE_RE.sub("\n\n", text)
    return "\n".join(line.strip() for line in text.split("\n")).strip()


def clean_text(value) -> str:
    text = to_unicode(value)
    text = normalize_encoding(text)
    text = strip_html(text)
    text = clean_wiki_noise(text)
    return text


def chunk(text: str, max_length: int = 200, overlap_ratio: float = 0.15) -> list[str]:
    words = [w for w in re.split(r'\s+', text) if w]
    if not words:
        return []

    overlap = int(max_length * overlap_ratio)
    step = max(1, max_length - overlap)

    chunks = []
    for i in range(0, len(words), step):
        chunks.append(' '.join(words[i:i + max_length]))
        if i + max_length >= len(words):
            break
    return chunks
