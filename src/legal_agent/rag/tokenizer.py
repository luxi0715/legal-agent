"""Chinese tokenizer for legal text using jieba."""

from functools import lru_cache
from pathlib import Path

import jieba

DICT_DIR = Path(__file__).parents[3] / "data" / "dicts"
LEGAL_TERMS = DICT_DIR / "legal_terms.txt"
STOPWORDS = DICT_DIR / "legal_stopwords.txt"


@lru_cache(maxsize=1)
def _init_jieba() -> set[str]:
    """Load custom dict and stopwords once. Returns stopwords set."""
    if LEGAL_TERMS.exists():
        jieba.load_userdict(str(LEGAL_TERMS))

    stopwords: set[str] = set()
    if STOPWORDS.exists():
        for line in STOPWORDS.read_text(encoding="utf-8").splitlines():
            word = line.strip()
            if word:
                stopwords.add(word)
    return stopwords


def tokenize(text: str, remove_stopwords: bool = True) -> list[str]:
    """Tokenize Chinese legal text into a list of tokens.

    Args:
        text: Raw Chinese text.
        remove_stopwords: Whether to filter common stopwords.

    Returns:
        List of tokens.
    """
    stopwords = _init_jieba()
    tokens = jieba.lcut(text)
    cleaned = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if remove_stopwords and tok in stopwords:
            continue
        cleaned.append(tok)
    return cleaned


def tokenize_for_tsvector(text: str) -> str:
    """Tokenize and return space-joined tokens for PostgreSQL tsvector.

    PostgreSQL tsvector treats whitespace as token boundary.
    By pre-tokenizing with jieba, we feed PG already-segmented words.
    """
    return " ".join(tokenize(text))
