import os
import json
import time
import logging
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path
import requests

logger = logging.getLogger("novelty_app")

@dataclass
class Doc:
    id: str
    source: str  # 'paper' or 'patent'
    title: str
    text: str
    year: int | None
    date: str | None
    url: str | None
    authors: list[str] | None
    venue: str | None
    extra: dict

USER_AGENT = {"User-Agent": "NoveltyChecker/1.0 (contact: admin@example.com)"}

# --- arXiv (public API; Atom) ---
ARXIV_API = "http://export.arxiv.org/api/query"

def fetch_arxiv(query: str, max_results: int = 100) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    start = 0
    step = min(100, max_results)
    while start < max_results:
        # Sort by lastUpdatedDate (descending) to get most recent papers first
        params = {"search_query": query, "start": start, "max_results": step, "sortBy": "lastUpdatedDate", "sortOrder": "descending"}
        r = requests.get(ARXIV_API, params=params, headers=USER_AGENT, timeout=30)
        r.raise_for_status()
        import xml.etree.ElementTree as ET
        root = ET.fromstring(r.content)
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        for entry in root.findall("atom:entry", ns):
            aid = entry.findtext("atom:id", default="", namespaces=ns)
            title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip().replace("\n", " ")
            summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
            updated = entry.findtext("atom:updated", default=None, namespaces=ns)
            link_el = entry.find("atom:link[@type='text/html']", ns)
            link = link_el.get("href") if link_el is not None else None
            authors = [a.findtext("atom:name", default="", namespaces=ns) for a in entry.findall("atom:author", ns)]
            year = None
            if updated:
                year = int(updated[:4])
            items.append(
                {
                    "id": aid,
                    "source": "paper",
                    "title": title,
                    "text": summary,
                    "year": year,
                    "date": updated,
                    "url": link,
                    "authors": authors,
                    "venue": "arXiv",
                    "extra": {},
                }
            )
        start += step
        time.sleep(0.5)
    return items

# --- IEEE Xplore (requires API key) ---
IEEE_ENDPOINT = "http://ieeexploreapi.ieee.org/api/v1/search/articles"

def fetch_ieee(query: str, max_results: int = 100) -> List[Dict[str, Any]]:
    api_key = os.getenv("IEEE_API_KEY")
    if not api_key:
        logger.info("IEEE_API_KEY not set; skipping IEEE fetch")
        return []
    items: List[Dict[str, Any]] = []
    page_size = min(200, max_results)
    params = {"apikey": api_key, "format": "json", "max_records": page_size, "start_record": 1, "querytext": query}
    r = requests.get(IEEE_ENDPOINT, params=params, headers=USER_AGENT, timeout=30)
    if r.status_code != 200:
        logger.warning("IEEE API error %s: %s", r.status_code, r.text[:200])
        return []
    data = r.json()
    for rec in data.get("articles", [])[:max_results]:
        year = int(rec.get("publication_year")) if rec.get("publication_year") else None
        items.append(
            {
                "id": f"ieee:{rec.get('doi') or rec.get('html_url')}",
                "source": "paper",
                "title": rec.get("title", ""),
                "text": rec.get("abstract", ""),
                "year": year,
                "date": rec.get("publication_date"),
                "url": rec.get("html_url"),
                "authors": [a.get("full_name") for a in rec.get("authors", {}).get("authors", []) if a.get("full_name")],
                "venue": rec.get("publication_title"),
                "extra": {"doi": rec.get("doi")},
            }
        )
    return items

# --- DOAJ (public API / OAI-PMH available). Use API v2 search if reachable. ---
DOAJ_SEARCH = "https://doaj.org/api/v2/search/articles/{}"

def fetch_doaj(query: str, max_results: int = 100) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    # If the v2 search endpoint is reachable, use it; otherwise skip silently.
    try:
        url = DOAJ_SEARCH.format(requests.utils.quote(query)) + f"?pageSize={min(100, max_results)}"
        r = requests.get(url, headers=USER_AGENT, timeout=30)
        if r.status_code != 200:
            logger.info("DOAJ API returned %s; skipping", r.status_code)
            return []
        data = r.json()
        for hit in data.get("results", [])[:max_results]:
            rec = hit.get("bibjson", {})
            year = None
            if rec.get("year"):
                try:
                    year = int(rec["year"])
                except Exception:
                    year = None
            items.append(
                {
                    "id": f"doaj:{hit.get('id')}",
                    "source": "paper",
                    "title": rec.get("title", ""),
                    "text": (rec.get("abstract") or ""),
                    "year": year,
                    "date": None,
                    "url": rec.get("link", [{}])[0].get("url") if rec.get("link") else None,
                    "authors": [a.get("name") for a in rec.get("author", []) if a.get("name")],
                    "venue": rec.get("journal", {}).get("title"),
                    "extra": {"keywords": rec.get("keywords")},
                }
            )
        return items
    except Exception:
        logger.exception("DOAJ fetch failed")
        return []

# --- Google Patents (no official free API; support third-party providers if keys present) ---
SEARCHAPI_IO = "https://www.searchapi.io/api/v1/search"
SERPAPI = "https://serpapi.com/search.json"

def fetch_google_patents(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    key = os.getenv("SEARCHAPI_KEY")
    provider = None
    if key:
        provider = "searchapi"
        params = {"engine": "google_patents", "q": query, "api_key": key}
        r = requests.get(SEARCHAPI_IO, params=params, headers=USER_AGENT, timeout=30)
        if r.status_code != 200:
            logger.warning("SearchAPI.io patents error %s", r.status_code)
            return []
        data = r.json()
        for res in (data.get("organic_results") or [])[:max_results]:
            items.append(
                {
                    "id": f"gpat:{res.get('patent_id') or res.get('title')}",
                    "source": "patent",
                    "title": res.get("title", ""),
                    "text": res.get("snippet", ""),
                    "year": int(res.get("publication_date", "")[:4]) if res.get("publication_date") and len(res.get("publication_date", "")) >= 4 else None,
                    "date": res.get("publication_date"),
                    "url": res.get("link"),
                    "authors": None,
                    "venue": "Google Patents",
                    "extra": res,
                }
            )
        return items
    key = os.getenv("SERPAPI_KEY")
    if key:
        provider = "serpapi"
        params = {"engine": "google_patents", "q": query, "api_key": key}
        r = requests.get(SERPAPI, params=params, headers=USER_AGENT, timeout=30)
        if r.status_code != 200:
            logger.warning("SerpAPI patents error %s", r.status_code)
            return []
        data = r.json()
        for res in (data.get("organic_results") or [])[:max_results]:
            items.append(
                {
                    "id": f"gpat:{res.get('patent_id') or res.get('title')}",
                    "source": "patent",
                    "title": res.get("title", ""),
                    "text": res.get("snippet", ""),
                    "year": int(res.get("publication_date", "")[:4]) if res.get("publication_date") and len(res.get("publication_date", "")) >= 4 else None,
                    "date": res.get("publication_date"),
                    "url": res.get("link"),
                    "authors": None,
                    "venue": "Google Patents",
                    "extra": res,
                }
            )
        return items
    logger.info("No Google Patents provider key set; skipping")
    return items

# --- Google Scholar (via SerpAPI) ---
def fetch_google_scholar(query: str, max_results: int = 50) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    key = os.getenv("SERPAPI_KEY")
    if not key:
        logger.info("SERPAPI_KEY not set; skipping Google Scholar fetch")
        return []

    try:
        params = {"engine": "google_scholar", "q": query, "api_key": key, "num": min(20, max_results)}
        r = requests.get(SERPAPI, params=params, headers=USER_AGENT, timeout=30)
        if r.status_code != 200:
            logger.warning("SerpAPI Scholar error %s", r.status_code)
            return []

        data = r.json()
        for res in (data.get("organic_results") or [])[:max_results]:
            # Extract year from publication_info if available
            year = None
            pub_info = res.get("publication_info", {})
            if pub_info.get("summary"):
                # Try to extract year from summary like "Author - 2023"
                import re
                year_match = re.search(r'\b(19|20)\d{2}\b', pub_info.get("summary", ""))
                if year_match:
                    year = int(year_match.group())

            items.append(
                {
                    "id": f"scholar:{res.get('result_id') or res.get('title')}",
                    "source": "paper",
                    "title": res.get("title", ""),
                    "text": res.get("snippet", ""),
                    "year": year,
                    "date": None,
                    "url": res.get("link"),
                    "authors": None,
                    "venue": "Google Scholar",
                    "extra": {"citation_count": res.get("inline_links", {}).get("cited_by", {}).get("total", 0)},
                }
            )
        return items
    except Exception as e:
        logger.exception("Google Scholar fetch failed")
        return []

# --- Aggregator ---

def collect_all_sources(query_terms: List[str], max_per_source: int, data_dir: Path) -> List[Dict[str, Any]]:
    query = " ".join(query_terms) or "novel computing"
    logger.info("Collecting sources for query: %s", query)
    all_docs: List[Dict[str, Any]] = []

    # arXiv
    try:
        all_docs += fetch_arxiv(query, max_results=max_per_source)
    except Exception:
        logger.exception("arXiv fetch failed")

    # IEEE
    try:
        all_docs += fetch_ieee(query, max_results=max_per_source)
    except Exception:
        logger.exception("IEEE fetch failed")

    # DOAJ
    try:
        all_docs += fetch_doaj(query, max_results=max_per_source)
    except Exception:
        logger.exception("DOAJ fetch failed")

    # Google Scholar (via SerpAPI)
    try:
        all_docs += fetch_google_scholar(query, max_results=max_per_source)
    except Exception:
        logger.exception("Google Scholar fetch failed")

    # Google Patents (via 3rd party)
    try:
        all_docs += fetch_google_patents(query, max_results=min(100, max_per_source))
    except Exception:
        logger.exception("Google Patents fetch failed")

    # Cache raw corpus for admin rebuilds
    try:
        import pandas as pd
        df = pd.DataFrame(all_docs)
        (data_dir / "corpus_cache.parquet").parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(data_dir / "corpus_cache.parquet", index=False)
    except Exception:
        logger.exception("Failed to cache corpus")

    return all_docs
