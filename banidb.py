"""BaniDB search client, used only when "Use BaniDB Only" is switched on.

API: https://api.banidb.com/v2/search/{query}?searchtype=N
- searchtype 2: full Gurmukhi word (Unicode), e.g. "ਕਰੋਧ"
- searchtype 3: English translation, e.g. "anger". Multi-word English phrases
  match only as an exact phrase and usually return nothing, so search single words.
"""

import json
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, List

BANIDB_SEARCH_URL = "https://api.banidb.com/v2/search/"
SEARCH_GURMUKHI_WORD = 2
SEARCH_ENGLISH_TRANSLATION = 3
REQUEST_TIMEOUT_SECONDS = 15


class BaniDBError(Exception):
    """BaniDB could not be reached or returned something unreadable."""


def search(query: str, searchtype: int, results: int = 5) -> List[Dict]:
    """Return up to `results` verse dicts from BaniDB for one query."""
    url = (
        BANIDB_SEARCH_URL
        + urllib.parse.quote(query.strip())
        + "?"
        + urllib.parse.urlencode({"searchtype": searchtype, "results": results})
    )
    request = urllib.request.Request(url, headers={"User-Agent": "GurbaniAI"})
    try:
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            data = json.load(response)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise BaniDBError(f"BaniDB search for {query!r} failed: {exc}") from exc
    return data.get("verses") or []


def search_many(english: List[str], gurmukhi: List[str], limit: int = 15) -> List[Dict]:
    """Run each query, keep the first `limit` distinct verses in query order."""
    queries = [(q, SEARCH_GURMUKHI_WORD) for q in gurmukhi]
    queries += [(q, SEARCH_ENGLISH_TRANSLATION) for q in english]
    seen = set()
    verses: List[Dict] = []
    for query, searchtype in queries:
        if not query.strip():
            continue
        for verse in search(query, searchtype):
            verse_id = verse.get("verseId")
            if verse_id in seen:
                continue
            seen.add(verse_id)
            verses.append(verse)
            if len(verses) >= limit:
                return verses
    return verses


def _english_translation(verse: Dict) -> str:
    en = (verse.get("translation") or {}).get("en") or {}
    # bdb (BaniDB), ms (Manmohan Singh), ssk (Sant Singh Khalsa)
    return en.get("bdb") or en.get("ms") or en.get("ssk") or ""


def format_passage(number: int, verse: Dict) -> str:
    """One verse as plain text: Gurmukhi, transliteration, translation, source."""
    source = verse.get("source") or {}
    writer = (verse.get("writer") or {}).get("english")
    raag = (verse.get("raag") or {}).get("english")
    where = [source.get("english") or "Unknown source"]
    if source.get("pageNo"):
        where.append(f"Ang {source['pageNo']}")
    if verse.get("lineNo"):
        where.append(f"line {verse['lineNo']}")
    if writer:
        where.append(writer)
    if raag:
        where.append(raag)  # already named, e.g. "Raag Maajh" or "Sohila"
    return (
        f"[{number}]\n"
        f"Gurmukhi: {(verse.get('verse') or {}).get('unicode', '')}\n"
        f"Transliteration: {(verse.get('transliteration') or {}).get('english', '')}\n"
        f"Translation: {_english_translation(verse)}\n"
        f"Source: {', '.join(where)}"
    )


def format_passages(verses: List[Dict]) -> str:
    return "\n\n".join(format_passage(i, v) for i, v in enumerate(verses, start=1))
