"""Paper search: arXiv and OpenAlex. Both are free; neither needs an API key.

OpenAlex gives requests without a key a small daily budget that is shared by everyone behind the same IP address
(about 100 searches a day in October 2026). A free OpenAlex key (OPENALEX_API_KEY in .env) gives its own budget.

Semantic Scholar was the first choice, but its unauthenticated API answered every request with 429 (rate limited)
during development, so OpenAlex is used instead: it indexes journals, conferences and arXiv, and returns abstracts
and citation counts.

Every HTTP response is cached on disk, so a run can be repeated exactly (and re-running costs no requests).
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import threading
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass

from .config import get_settings

UA = "litreview-portfolio-project/1.0 (research assistant demo)"
log = logging.getLogger("litreview.sources")


@dataclass
class Paper:
    id: str                 # "arxiv:2301.01234" or "openalex:W123"
    title: str
    abstract: str
    year: int | None
    authors: list[str]
    venue: str
    url: str
    citations: int | None
    source: str

    @property
    def key(self) -> str:
        """Normalised title, used to merge the same paper found by both sources."""
        return re.sub(r"[^a-z0-9]", "", self.title.lower())[:80]

    def to_dict(self) -> dict:
        return asdict(self)


class _HttpCache:
    def __init__(self):
        path = get_settings().cache_dir / "http_cache.sqlite"
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.execute("create table if not exists http(url text primary key, body text, ts real)")
        self.lock = threading.Lock()
        self.last: dict[str, float] = {}

    def get(self, url: str, min_interval: float, host: str, fetch_url: str | None = None) -> str:
        """`url` is the cache key; `fetch_url` (default `url`) is what is requested, e.g. with an API key added."""
        with self.lock:
            row = self.db.execute("select body from http where url = ?", (url,)).fetchone()
        if row:
            return row[0]
        for attempt in range(4):
            with self.lock:                      # be polite: one request per `min_interval` per host
                wait = self.last.get(host, 0) + min_interval - time.monotonic()
                self.last[host] = time.monotonic() + max(wait, 0)
            if wait > 0:
                time.sleep(wait)
            try:
                req = urllib.request.Request(fetch_url or url, headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=30) as r:
                    body = r.read().decode("utf-8")
                with self.lock:
                    self.db.execute("insert or replace into http values (?,?,?)", (url, body, time.time()))
                    self.db.commit()
                return body
            except Exception:  # noqa: BLE001 - network hiccup or 429: back off and retry
                if attempt == 3:
                    raise
                time.sleep(3 * (attempt + 1))
        raise RuntimeError("unreachable")


_cache: _HttpCache | None = None


def _http() -> _HttpCache:
    global _cache
    if _cache is None:
        _cache = _HttpCache()
    return _cache


def search_arxiv(query: str, n: int = 8) -> list[Paper]:
    words = [w for w in re.findall(r"[A-Za-z0-9\-]+", query) if len(w) > 2][:8]
    q = " AND ".join(f"all:{w}" for w in words)
    url = "https://export.arxiv.org/api/query?" + urllib.parse.urlencode(
        {"search_query": q, "max_results": n, "sortBy": "relevance"})
    xml = _http().get(url, 3.0, "arxiv")          # arXiv asks for at most one request every 3 seconds
    ns = {"a": "http://www.w3.org/2005/Atom"}
    out = []
    for e in ET.fromstring(xml).findall("a:entry", ns):
        aid = e.findtext("a:id", "", ns).rsplit("/abs/", 1)[-1]
        aid = re.sub(r"v\d+$", "", aid)
        title = " ".join(e.findtext("a:title", "", ns).split())
        abstract = " ".join(e.findtext("a:summary", "", ns).split())
        year = int(e.findtext("a:published", "0000", ns)[:4]) or None
        authors = [a.findtext("a:name", "", ns) for a in e.findall("a:author", ns)]
        out.append(Paper(f"arxiv:{aid}", title, abstract, year, authors, "arXiv", f"https://arxiv.org/abs/{aid}", None, "arxiv"))
    return out


def _abstract_from_index(inv: dict | None) -> str:
    if not inv:
        return ""
    pos = sorted((p, w) for w, ps in inv.items() for p in ps)
    return " ".join(w for _, w in pos)


def clean_query(query: str) -> str:
    """Letters, digits, hyphens and spaces only: OpenAlex answers 400 to some punctuation (e.g. a trailing "?")."""
    return " ".join(re.sub(r"[^A-Za-z0-9\- ]", " ", query).split())


def search_openalex(query: str, n: int = 8) -> list[Paper]:
    url = "https://api.openalex.org/works?" + urllib.parse.urlencode(
        {"search": clean_query(query), "per-page": n, "filter": "has_abstract:true",
         "select": "id,display_name,abstract_inverted_index,publication_year,authorships,primary_location,cited_by_count,doi"})
    key = os.getenv("OPENALEX_API_KEY", "").strip()      # kept out of the cache key, so cached replies still match
    fetch = f"{url}&{urllib.parse.urlencode({'api_key': key})}" if key else None
    data = json.loads(_http().get(url, 0.2, "openalex", fetch))
    out = []
    for w in data.get("results", []):
        loc = (w.get("primary_location") or {})
        venue = ((loc.get("source") or {}).get("display_name")) or ""
        out.append(Paper(
            f"openalex:{w['id'].rsplit('/', 1)[-1]}", " ".join((w.get("display_name") or "").split()),
            _abstract_from_index(w.get("abstract_inverted_index")), w.get("publication_year"),
            [a["author"]["display_name"] for a in (w.get("authorships") or [])[:6] if a.get("author")], venue,
            w.get("doi") or loc.get("landing_page_url") or w["id"], w.get("cited_by_count"), "openalex"))
    return out


def search(query: str, n_each: int = 8) -> list[Paper]:
    """Both sources, merged by normalised title (OpenAlex metadata kept when both have a paper: it has citation counts)."""
    merged: dict[str, Paper] = {}
    found = []
    for name, fn in (("openalex", search_openalex), ("arxiv", search_arxiv)):
        try:
            found += fn(query, n_each)
        except Exception as e:  # noqa: BLE001 - one source failing must not stop the review
            log.warning("%s search failed for %r: %s", name, query, e)
    for p in found:
        if len(p.abstract) < 200 or not p.title:
            continue                  # no abstract = nothing to ground a claim in
        merged.setdefault(p.key, p)
    return list(merged.values())


def cache_key(*parts) -> str:
    return hashlib.sha256(json.dumps(parts).encode()).hexdigest()
