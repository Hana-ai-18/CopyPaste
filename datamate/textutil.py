"""Text normalisation shared by the scorer, the voter and the sandbox helpers."""
from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def norm_string(value) -> str:
    """Official `string` normalisation of the challenge.

    1. Unicode NFC  2. lowercase  3. every character that is not a letter, a
    digit or whitespace becomes a space  4. strip and collapse whitespace.
    Diacritics are NOT removed.
    """
    s = nfc(str(value)).lower()
    s = "".join(ch if (ch.isalnum() or ch.isspace()) else " " for ch in s)
    return _WS.sub(" ", s).strip()


def strip_accents(value) -> str:
    """Remove Vietnamese diacritics (and map đ -> d). For *matching helpers only*."""
    s = nfc(str(value)).replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", s)
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def vi_key(value) -> str:
    """Accent-insensitive, case-insensitive key (for fuzzy lookups and retrieval)."""
    return norm_string(strip_accents(value))


def vi_tokens(value) -> list[str]:
    k = vi_key(value)
    return k.split() if k else []
