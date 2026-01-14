"""
Best-effort public incident dataset builder for the AVS risk project.

Goal
----
Produce a real (non-mock) dataset of security/incident events from public sources that can be
joined/aggregated later into AVS-week labels or used as qualitative evidence.

Design principles
-----------------
- Prefer stable, machine-readable sources (APIs, RSS/Atom, GitHub APIs).
- Keep raw source snapshots for reproducibility.
- Be explicit about "best effort": some sources may be unavailable or change format.

Outputs
-------
- data/derived/incidents_public.csv
- data/derived/incidents_raw/*.json (optional)

Usage
-----
python -m src.incidents_public_dataset --since-days 365
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Allow running as a script: `python src/incidents_public_dataset.py`
# (When executed this way, `src` isn't automatically on sys.path.)
_REPO_ROOT_FOR_SCRIPT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_FOR_SCRIPT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_SCRIPT))

from src.global_conf import CONF
from src.github_incidents_util import GitHubClient, GitHubIncidentsUtil


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
DERIVED_DIR = DATA_DIR / "derived"
RAW_DIR = DERIVED_DIR / "incidents_raw"

ECOSYSTEM_KEYWORDS_BASE = [
    # EigenLayer / restaking domain terms
    "eigenlayer",
    "restaking",
    "restake",
    "avs",
    "actively validated service",
    "operator",
    "slashing",
    "delegation manager",
    "eigenpod",
    "eigenpod manager",
    "avs directory",
    "eigendata",
    "eigend a",
    "eigen da",
    "eigenda",
]

INCIDENT_KEYWORDS_BASE = [
    "incident",
    "postmortem",
    "outage",
    "exploit",
    "hack",
    "attack",
    "drain",
    "vulnerability",
    "security",
    "cve",
    "ghsa",
]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_dt(x: Any) -> Optional[datetime]:
    """
    Best-effort datetime parser for heterogeneous feeds.
    Returns timezone-aware UTC datetimes.
    """
    if x is None:
        return None
    if isinstance(x, datetime):
        if x.tzinfo is None:
            return x.replace(tzinfo=timezone.utc)
        return x.astimezone(timezone.utc)
    s = str(x).strip()
    if not s:
        return None
    # ISO-8601
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        pass
    # numeric unix seconds / ms
    try:
        if re.fullmatch(r"\d{10,13}", s):
            ts = int(s)
            if ts > 10**12:  # ms
                ts = ts // 1000
            return datetime.fromtimestamp(ts, tz=timezone.utc)
    except Exception:
        pass
    return None


def _stable_id(*parts: str) -> str:
    """
    Deterministic, short ID suitable for CSV.
    """
    base = "|".join([p.strip() for p in parts if isinstance(p, str) and p.strip()])
    h = hashlib.sha1(base.encode("utf-8")).hexdigest()  # nosec - not for security
    return h[:12]


def _safe_json_dumps(obj: Any) -> str:
    try:
        return json.dumps(obj, ensure_ascii=False, sort_keys=True)
    except Exception:
        return json.dumps(str(obj), ensure_ascii=False)


@dataclass(frozen=True, slots=True)
class Incident:
    incident_id: str
    source: str
    source_url: str
    source_type: str  # api|rss|github|scrape
    title: str
    occurred_at: Optional[datetime]
    discovered_at: datetime
    protocol: Optional[str]
    chain: Optional[str]
    amount_usd: Optional[float]
    category: Optional[str]
    description: Optional[str]
    tags: List[str]
    raw: Dict[str, Any]

    def to_row(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "source": self.source,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "title": self.title,
            "occurred_at": self.occurred_at.isoformat() if self.occurred_at else "",
            "discovered_at": self.discovered_at.isoformat(),
            "protocol": self.protocol or "",
            "chain": self.chain or "",
            "amount_usd": self.amount_usd if self.amount_usd is not None else "",
            "category": self.category or "",
            "description": self.description or "",
            "tags": ",".join(sorted(set([t for t in self.tags if t]))),
            "raw_json": _safe_json_dumps(self.raw),
        }


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _contains_any(text: str, needles: Iterable[str]) -> bool:
    t = _norm(text)
    for n in needles:
        nn = _norm(n)
        if not nn or len(nn) < 3:
            continue
        if nn in t:
            return True
    return False


def _load_ecosystem_terms(logger: logging.Logger) -> List[str]:
    """
    Build a best-effort allowlist of EigenLayer/restaking ecosystem terms from existing repo data.
    """
    terms: set[str] = set(ECOSYSTEM_KEYWORDS_BASE)

    # AVS names from u--1 metadata (best signal for "what is an AVS in our dataset")
    u1_path = DATA_DIR / "u1_latest_avs_balances.json"
    if u1_path.exists():
        try:
            obj = json.loads(u1_path.read_text(encoding="utf-8"))
            if isinstance(obj, dict):
                for _, payload in obj.items():
                    meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
                    if isinstance(meta, dict):
                        name = meta.get("name")
                        if isinstance(name, str) and name.strip():
                            terms.add(name.strip())
        except Exception as e:
            logger.warning(f"Failed loading u1_latest_avs_balances names: {type(e).__name__}: {e}")

    # Small, high-signal set of restaking/LRT ecosystem names (for discovery in public sources).
    for x in [
        "ether.fi",
        "etherfi",
        "renzo",
        "kelp",
        "kelpdao",
        "swell",
        "puffer",
        "stakewise",
        "stader",
        "ssv",
        "lido",
        "rocket pool",
        "bedrock",
        "zoth",
    ]:
        terms.add(x)

    return sorted(terms)


def _is_restaking_related(inc: "Incident", ecosystem_terms: List[str]) -> bool:
    blob = " | ".join(
        [
            inc.source,
            inc.source_url,
            inc.title,
            inc.protocol or "",
            inc.chain or "",
            inc.category or "",
            inc.description or "",
            " ".join(inc.tags or []),
        ]
    )
    return _contains_any(blob, ecosystem_terms)


def _is_incident_like(inc: "Incident") -> bool:
    """
    For noisy sources (notably GitHub), require explicit incident/security keywords.
    """
    # Important: do NOT include `category` in this heuristic.
    # If `category` contains a generic token like "incident", it can accidentally make every
    # GitHub row pass this filter.
    blob = " | ".join([inc.title or "", inc.description or "", " ".join(inc.tags or [])])
    return _contains_any(blob, INCIDENT_KEYWORDS_BASE)


class HttpClient:
    def __init__(
        self,
        *,
        timeout_secs: int = CONF.HTTP_TIMEOUT_SECS,
        max_retries: int = CONF.HTTP_MAX_RETRIES,
        backoff_secs: float = CONF.HTTP_BACKOFF_SECS,
        user_agent: str = "sc_review_incidents_public_dataset",
    ) -> None:
        self.timeout_secs = int(timeout_secs)
        self.user_agent = user_agent
        s = requests.Session()
        retry = Retry(
            total=max(0, int(max_retries)),
            backoff_factor=float(backoff_secs),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        self._session = s

    def get(self, url: str, *, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> requests.Response:
        h = {"User-Agent": self.user_agent, "Accept": "*/*"}
        if headers:
            h.update(headers)
        return self._session.get(url, headers=h, params=params, timeout=self.timeout_secs)

    def get_json(self, url: str, *, headers: Optional[Dict[str, str]] = None, params: Optional[Dict[str, Any]] = None) -> Any:
        resp = self.get(url, headers=headers, params=params)
        resp.raise_for_status()
        return resp.json()


class DeFiLlamaHacksCollector:
    """
    DeFiLlama maintains a public hacks dataset (best-effort endpoint discovery).
    """

    # Common endpoints observed in the wild (we try several).
    CANDIDATE_URLS = [
        "https://api.llama.fi/hacks",
        "https://api.llama.fi/hack",
        "https://api.llama.fi/hacks?sort=desc",
    ]

    def __init__(self, http: HttpClient, logger: logging.Logger) -> None:
        self.http = http
        self.logger = logger

    def _fetch(self) -> Tuple[str, Any]:
        last_err: Optional[str] = None
        for url in self.CANDIDATE_URLS:
            try:
                data = self.http.get_json(url)
                return url, data
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                continue
        raise RuntimeError(f"Failed to fetch DeFiLlama hacks from candidate URLs. Last error: {last_err}")

    def collect(self, *, since_dt: datetime, write_raw: bool) -> List[Incident]:
        url, data = self._fetch()
        if write_raw:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            (RAW_DIR / "defillama_hacks.json").write_text(json.dumps({"fetched_from": url, "data": data}, indent=2), encoding="utf-8")

        # Format may be either list[dict] or dict with "hacks" key.
        rows: List[Dict[str, Any]] = []
        if isinstance(data, list):
            rows = [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict):
            hacks = data.get("hacks") or data.get("data") or data.get("events")
            if isinstance(hacks, list):
                rows = [r for r in hacks if isinstance(r, dict)]

        incidents: List[Incident] = []
        now = _utc_now()
        for r in rows:
            # Common fields: name/protocol, date, amount, chain, classification
            title = str(r.get("name") or r.get("protocol") or r.get("project") or r.get("title") or "DeFi incident").strip()
            source_url = str(r.get("link") or r.get("url") or "https://defillama.com/hacks").strip()
            occurred = _parse_dt(r.get("date") or r.get("timestamp") or r.get("time") or r.get("reportedAt"))
            if occurred and occurred < since_dt:
                continue

            amount_usd = None
            for k in ("amount", "amount_usd", "amountUsd", "loss", "funds_lost", "fundsLost"):
                if k in r:
                    try:
                        amount_usd = float(r.get(k))
                    except Exception:
                        amount_usd = None
                    break

            protocol = r.get("name") or r.get("protocol") or r.get("project")
            if protocol is not None:
                protocol = str(protocol).strip() or None

            chain = r.get("chain") or r.get("chains") or r.get("network")
            if isinstance(chain, list):
                chain = ",".join([str(x) for x in chain if x is not None])
            chain = str(chain).strip() if chain is not None else None
            chain = chain or None

            category = r.get("classification") or r.get("category") or r.get("type")
            category = str(category).strip() if category is not None else None
            category = category or None

            desc = r.get("description") or r.get("summary")
            desc = str(desc).strip() if desc is not None else None
            desc = desc or None

            tags: List[str] = []
            for k in ("attackType", "attack_type", "technique", "method"):
                v = r.get(k)
                if isinstance(v, str) and v.strip():
                    tags.append(v.strip())

            incident_id = _stable_id("defillama", title, (occurred.isoformat() if occurred else ""), source_url)
            incidents.append(
                Incident(
                    incident_id=incident_id,
                    source="defillama_hacks",
                    source_url=source_url,
                    source_type="api",
                    title=title,
                    occurred_at=occurred,
                    discovered_at=now,
                    protocol=protocol,
                    chain=chain,
                    amount_usd=amount_usd,
                    category=category,
                    description=desc,
                    tags=tags,
                    raw=r,
                )
            )
        self.logger.info(f"DeFiLlama hacks collected: {len(incidents)} (since {since_dt.date().isoformat()})")
        return incidents


class EigenLayerBlogCollector:
    """
    Best-effort collector for EigenLayer official posts.
    Prefer RSS/Atom if available; otherwise we can fall back to a sitemap-like endpoint if discovered.
    """

    CANDIDATE_FEEDS = [
        "https://www.eigenlayer.xyz/rss.xml",
        "https://www.eigenlayer.xyz/feed.xml",
        "https://www.eigenlayer.xyz/blog/rss.xml",
        "https://www.eigenlayer.xyz/blog/feed.xml",
        "https://www.eigenlayer.xyz/blog/rss",
        "https://www.eigenlayer.xyz/blog/feed",
    ]

    def __init__(self, http: HttpClient, logger: logging.Logger) -> None:
        self.http = http
        self.logger = logger

    def _fetch_feed_xml(self) -> Tuple[str, str]:
        last_err: Optional[str] = None
        for url in self.CANDIDATE_FEEDS:
            try:
                resp = self.http.get(url, headers={"Accept": "application/rss+xml, application/atom+xml, text/xml;q=0.9, */*;q=0.8"})
                if resp.status_code >= 400:
                    last_err = f"HTTP {resp.status_code} from {url}"
                    continue
                text = resp.text or ""
                # Heuristic: must look like XML feed
                if "<rss" in text.lower() or "<feed" in text.lower():
                    return url, text
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                continue
        raise RuntimeError(f"Failed to fetch EigenLayer RSS/Atom feed from candidates. Last error: {last_err}")

    def _parse_feed(self, xml_text: str) -> List[Dict[str, Any]]:
        # Avoid adding deps (beautifulsoup); use ElementTree.
        import xml.etree.ElementTree as ET

        # Strip invalid control chars sometimes present in feeds
        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", xml_text)
        root = ET.fromstring(cleaned)

        def text_of(el: Optional[ET.Element], *names: str) -> str:
            if el is None:
                return ""
            for n in names:
                child = el.find(n)
                if child is not None and (child.text or "").strip():
                    return (child.text or "").strip()
            return ""

        items: List[ET.Element] = []
        if root.tag.lower().endswith("rss") or root.find("channel") is not None:
            ch = root.find("channel")
            if ch is not None:
                items = list(ch.findall("item"))
        else:
            # Atom: entries
            items = list(root.findall("{http://www.w3.org/2005/Atom}entry")) or list(root.findall("entry"))

        out: List[Dict[str, Any]] = []
        for it in items:
            title = text_of(it, "title", "{http://www.w3.org/2005/Atom}title")
            link = ""
            # RSS link element
            link = text_of(it, "link")
            if not link:
                # Atom link attribute
                for l in it.findall("{http://www.w3.org/2005/Atom}link") + it.findall("link"):
                    href = l.attrib.get("href") if hasattr(l, "attrib") else None
                    if href:
                        link = href
                        break
            pub = text_of(it, "pubDate", "{http://www.w3.org/2005/Atom}published", "{http://www.w3.org/2005/Atom}updated", "published", "updated")
            out.append({"title": title, "link": link, "published": pub})
        return out

    def collect(self, *, since_dt: datetime, write_raw: bool) -> List[Incident]:
        url, xml_text = self._fetch_feed_xml()
        entries = self._parse_feed(xml_text)
        if write_raw:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            (RAW_DIR / "eigenlayer_blog_feed.json").write_text(
                json.dumps({"fetched_from": url, "entries": entries}, indent=2),
                encoding="utf-8",
            )

        now = _utc_now()
        incidents: List[Incident] = []
        for e in entries:
            title = str(e.get("title") or "EigenLayer blog post").strip()
            link = str(e.get("link") or "https://www.eigenlayer.xyz/blog").strip()
            pub = _parse_dt(e.get("published"))
            if pub and pub < since_dt:
                continue
            # Treat as "announcement" (not necessarily an incident) but useful primary evidence.
            incident_id = _stable_id("eigenlayer_blog", title, (pub.isoformat() if pub else ""), link)
            incidents.append(
                Incident(
                    incident_id=incident_id,
                    source="eigenlayer_blog",
                    source_url=link,
                    source_type="rss",
                    title=title,
                    occurred_at=pub,
                    discovered_at=now,
                    protocol="EigenLayer",
                    chain="Ethereum",
                    amount_usd=None,
                    category="announcement",
                    description=None,
                    tags=["eigenlayer", "blog"],
                    raw=e,
                )
            )

        self.logger.info(f"EigenLayer blog entries collected: {len(incidents)} (since {since_dt.date().isoformat()})")
        return incidents


class EigenLayerSitemapCollector:
    """
    Best-effort fallback for EigenLayer official posts:
    - Try sitemap.xml
    - Extract blog URLs
    - Fetch pages and parse JSON-LD / OG meta for title + date
    """

    SITEMAP_URLS = [
        "https://www.eigenlayer.xyz/sitemap.xml",
        "https://www.eigenlayer.xyz/sitemap-0.xml",
    ]

    def __init__(self, http: HttpClient, logger: logging.Logger) -> None:
        self.http = http
        self.logger = logger

    def _fetch_sitemap(self) -> Tuple[str, str]:
        last_err: Optional[str] = None
        for url in self.SITEMAP_URLS:
            try:
                resp = self.http.get(url, headers={"Accept": "application/xml, text/xml;q=0.9, */*;q=0.8"})
                if resp.status_code >= 400:
                    last_err = f"HTTP {resp.status_code} from {url}"
                    continue
                text = resp.text or ""
                if "<urlset" in text.lower() or "<sitemapindex" in text.lower():
                    return url, text
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                continue
        raise RuntimeError(f"Failed to fetch EigenLayer sitemap. Last error: {last_err}")

    def _extract_urls(self, xml_text: str) -> List[Tuple[str, Optional[datetime]]]:
        import xml.etree.ElementTree as ET

        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", xml_text)
        root = ET.fromstring(cleaned)
        out: List[Tuple[str, Optional[datetime]]] = []

        # urlset entries
        for url_el in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url") + root.findall(".//url"):
            loc_el = url_el.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc") or url_el.find("loc")
            if loc_el is None or not (loc_el.text or "").strip():
                continue
            loc = (loc_el.text or "").strip()
            last_el = url_el.find("{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod") or url_el.find("lastmod")
            last = _parse_dt((last_el.text or "").strip()) if last_el is not None else None
            out.append((loc, last))

        # sitemapindex -> nested sitemaps
        if root.tag.endswith("sitemapindex") or root.find(".//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap") is not None:
            for sm_el in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap") + root.findall(".//sitemap"):
                loc_el = sm_el.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc") or sm_el.find("loc")
                if loc_el is None or not (loc_el.text or "").strip():
                    continue
                loc = (loc_el.text or "").strip()
                try:
                    resp = self.http.get(loc, headers={"Accept": "application/xml, text/xml;q=0.9, */*;q=0.8"})
                    if resp.status_code >= 400:
                        continue
                    out.extend(self._extract_urls(resp.text or ""))
                except Exception:
                    continue

        return out

    def _parse_page_meta(self, html: str) -> Tuple[str, Optional[datetime], Optional[str]]:
        # JSON-LD first
        for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.I | re.S):
            blob = m.group(1).strip()
            if not blob:
                continue
            try:
                data = json.loads(blob)
            except Exception:
                continue
            candidates = data if isinstance(data, list) else [data]
            for d in candidates:
                if not isinstance(d, dict):
                    continue
                title = d.get("headline") or d.get("name")
                dt = _parse_dt(d.get("datePublished") or d.get("dateModified"))
                desc = d.get("description")
                if title:
                    return str(title).strip(), dt, str(desc).strip() if isinstance(desc, str) else None

        # OG meta fallback
        og_title = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        title = og_title.group(1).strip() if og_title else "EigenLayer post"
        pub = None
        og_time = re.search(
            r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']', html, re.I
        )
        if og_time:
            pub = _parse_dt(og_time.group(1))
        desc = None
        og_desc = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        if og_desc:
            desc = og_desc.group(1).strip()
        return title, pub, desc

    def collect(self, *, since_dt: datetime, write_raw: bool, max_pages: int = 200) -> List[Incident]:
        url, xml_text = self._fetch_sitemap()
        urls = self._extract_urls(xml_text)

        # Keep blog URLs only
        blog_urls = [(u, lm) for (u, lm) in urls if "/blog/" in u]
        blog_urls.sort(key=lambda x: (x[1] or datetime(1970, 1, 1, tzinfo=timezone.utc)), reverse=True)

        now = _utc_now()
        incidents: List[Incident] = []
        raw_entries: List[Dict[str, Any]] = []

        for (page_url, lastmod) in blog_urls[: max(1, int(max_pages))]:
            if lastmod and lastmod < since_dt:
                continue
            try:
                resp = self.http.get(page_url, headers={"Accept": "text/html, */*;q=0.8"})
                if resp.status_code >= 400:
                    continue
                html = resp.text or ""
                title, pub, desc = self._parse_page_meta(html)
                occurred = pub or lastmod
                if occurred and occurred < since_dt:
                    continue

                entry = {"title": title, "url": page_url, "published": (occurred.isoformat() if occurred else ""), "description": desc}
                raw_entries.append(entry)
                incidents.append(
                    Incident(
                        incident_id=_stable_id("eigenlayer_sitemap", title, (occurred.isoformat() if occurred else ""), page_url),
                        source="eigenlayer_blog",
                        source_url=page_url,
                        source_type="scrape",
                        title=title,
                        occurred_at=occurred,
                        discovered_at=now,
                        protocol="EigenLayer",
                        chain="Ethereum",
                        amount_usd=None,
                        category="announcement",
                        description=desc,
                        tags=["eigenlayer", "blog"],
                        raw=entry,
                    )
                )
            except Exception:
                continue

        if write_raw:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            (RAW_DIR / "eigenlayer_sitemap_blog.json").write_text(
                json.dumps({"fetched_from": url, "entries": raw_entries}, indent=2),
                encoding="utf-8",
            )

        self.logger.info(f"EigenLayer sitemap blog entries collected: {len(incidents)} (since {since_dt.date().isoformat()})")
        return incidents


class RektSitemapCollector:
    """
    REKT is a curated incident writeup site. We use its sitemap as a best-effort discovery tool.
    We only include entries that look EigenLayer/restaking-related via keyword filtering.
    """

    SITEMAP_URLS = [
        "https://rekt.news/sitemap.xml",
        "https://rekt.news/sitemap_index.xml",
    ]

    def __init__(self, http: HttpClient, logger: logging.Logger) -> None:
        self.http = http
        self.logger = logger

    def _fetch_sitemap(self) -> Tuple[str, str]:
        last_err: Optional[str] = None
        for url in self.SITEMAP_URLS:
            try:
                resp = self.http.get(url, headers={"Accept": "application/xml, text/xml;q=0.9, */*;q=0.8"})
                if resp.status_code >= 400:
                    last_err = f"HTTP {resp.status_code} from {url}"
                    continue
                text = resp.text or ""
                if "<urlset" in text.lower() or "<sitemapindex" in text.lower():
                    return url, text
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                continue
        raise RuntimeError(f"Failed to fetch REKT sitemap. Last error: {last_err}")

    def _extract_urls(self, xml_text: str) -> List[Tuple[str, Optional[datetime]]]:
        import xml.etree.ElementTree as ET

        cleaned = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", xml_text)
        root = ET.fromstring(cleaned)
        out: List[Tuple[str, Optional[datetime]]] = []

        for url_el in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}url") + root.findall(".//url"):
            loc_el = url_el.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc") or url_el.find("loc")
            if loc_el is None or not (loc_el.text or "").strip():
                continue
            loc = (loc_el.text or "").strip()
            last_el = url_el.find("{http://www.sitemaps.org/schemas/sitemap/0.9}lastmod") or url_el.find("lastmod")
            last = _parse_dt((last_el.text or "").strip()) if last_el is not None else None
            out.append((loc, last))

        if root.tag.endswith("sitemapindex") or root.find(".//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap") is not None:
            for sm_el in root.findall(".//{http://www.sitemaps.org/schemas/sitemap/0.9}sitemap") + root.findall(".//sitemap"):
                loc_el = sm_el.find("{http://www.sitemaps.org/schemas/sitemap/0.9}loc") or sm_el.find("loc")
                if loc_el is None or not (loc_el.text or "").strip():
                    continue
                loc = (loc_el.text or "").strip()
                try:
                    resp = self.http.get(loc, headers={"Accept": "application/xml, text/xml;q=0.9, */*;q=0.8"})
                    if resp.status_code >= 400:
                        continue
                    out.extend(self._extract_urls(resp.text or ""))
                except Exception:
                    continue

        return out

    def _parse_page_meta(self, html: str) -> Tuple[str, Optional[datetime], Optional[str]]:
        for m in re.finditer(r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.I | re.S):
            blob = m.group(1).strip()
            if not blob:
                continue
            try:
                data = json.loads(blob)
            except Exception:
                continue
            candidates = data if isinstance(data, list) else [data]
            for d in candidates:
                if not isinstance(d, dict):
                    continue
                title = d.get("headline") or d.get("name")
                dt = _parse_dt(d.get("datePublished") or d.get("dateModified"))
                desc = d.get("description")
                if title:
                    return str(title).strip(), dt, str(desc).strip() if isinstance(desc, str) else None

        og_title = re.search(r'<meta[^>]+property=["\']og:title["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        title = og_title.group(1).strip() if og_title else "REKT incident"
        pub = None
        og_time = re.search(
            r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']', html, re.I
        )
        if og_time:
            pub = _parse_dt(og_time.group(1))
        og_desc = re.search(r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\']([^"\']+)["\']', html, re.I)
        desc = og_desc.group(1).strip() if og_desc else None
        return title, pub, desc

    def collect(
        self,
        *,
        since_dt: datetime,
        write_raw: bool,
        ecosystem_terms: List[str],
        max_pages: int = 250,
    ) -> List[Incident]:
        url, xml_text = self._fetch_sitemap()
        urls = self._extract_urls(xml_text)

        post_urls = [(u, lm) for (u, lm) in urls if u.startswith("https://rekt.news/") and "/tag/" not in u and "/author/" not in u]
        post_urls.sort(key=lambda x: (x[1] or datetime(1970, 1, 1, tzinfo=timezone.utc)), reverse=True)

        now = _utc_now()
        incidents: List[Incident] = []
        raw_entries: List[Dict[str, Any]] = []

        for (page_url, lastmod) in post_urls[: max(1, int(max_pages))]:
            if lastmod and lastmod < since_dt:
                continue
            # quick pre-filter by URL slug
            if not _contains_any(page_url, ecosystem_terms):
                continue
            try:
                resp = self.http.get(page_url, headers={"Accept": "text/html, */*;q=0.8"})
                if resp.status_code >= 400:
                    continue
                html = resp.text or ""
                title, pub, desc = self._parse_page_meta(html)
                occurred = pub or lastmod
                if occurred and occurred < since_dt:
                    continue

                # Ensure content is actually ecosystem-related
                if not _contains_any(" | ".join([title, desc or "", page_url]), ecosystem_terms):
                    continue

                entry = {"title": title, "url": page_url, "published": (occurred.isoformat() if occurred else ""), "description": desc}
                raw_entries.append(entry)
                incidents.append(
                    Incident(
                        incident_id=_stable_id("rekt", title, (occurred.isoformat() if occurred else ""), page_url),
                        source="rekt",
                        source_url=page_url,
                        source_type="scrape",
                        title=title,
                        occurred_at=occurred,
                        discovered_at=now,
                        protocol=None,
                        chain=None,
                        amount_usd=None,
                        category="incident_writeup",
                        description=desc,
                        tags=["rekt"],
                        raw=entry,
                    )
                )
            except Exception:
                continue

        if write_raw:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            (RAW_DIR / "rekt_entries.json").write_text(
                json.dumps({"fetched_from": url, "entries": raw_entries}, indent=2),
                encoding="utf-8",
            )

        self.logger.info(f"REKT entries collected (filtered): {len(incidents)} (since {since_dt.date().isoformat()})")
        return incidents


class GitHubSignalsCollector:
    """
    Convert repo-scoped GitHub "incident-like" issues/PRs into incident candidates.
    (These are not confirmed exploits, but are extremely useful for primary evidence + change points.)
    """

    def __init__(self, logger: logging.Logger) -> None:
        self.logger = logger
        self.util = GitHubIncidentsUtil()
        self.client = GitHubClient()

    def collect(self, *, since_days: int, repos: Optional[List[str]], write_raw: bool) -> List[Incident]:
        # Use repo discovery if not provided
        if repos is None:
            base = [
                "Layr-Labs/eigenlayer-contracts",
                "Layr-Labs/eigensdk-go",
                "Layr-Labs/eigenda",
            ]
            discovered = self.util.discover_repos_from_u1_metadata()
            repos = sorted(set(base + discovered))

        # Reuse existing util to keep behavior consistent (rate limit handling, etc.)
        out_path = self.util.run(repos, since_days=since_days, out_path=DATA_DIR / "github_case_studies.json")
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        if write_raw:
            RAW_DIR.mkdir(parents=True, exist_ok=True)
            (RAW_DIR / "github_case_studies.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

        now = _utc_now()
        since_dt = now - timedelta(days=max(0, int(since_days)))

        incidents: List[Incident] = []
        for bundle in payload.get("repos", []) or []:
            repo = bundle.get("repo")
            for it in bundle.get("incident_issues_prs", []) or []:
                title = str(it.get("title") or "").strip()
                url = str(it.get("html_url") or "").strip()
                created = _parse_dt(it.get("created_at"))
                updated = _parse_dt(it.get("updated_at"))
                occurred = created or updated
                if occurred and occurred < since_dt:
                    continue
                labels = it.get("labels") or []
                tags = [str(x).strip() for x in labels if isinstance(x, str)]
                tags.extend(["github", "issue_pr", str(repo or "").strip()])
                incident_id = _stable_id("github_issue", str(repo or ""), title, (occurred.isoformat() if occurred else ""), url)
                incidents.append(
                    Incident(
                        incident_id=incident_id,
                        source="github_search",
                        source_url=url or f"https://github.com/{repo}" if repo else "https://github.com",
                        source_type="github",
                        title=title or "GitHub issue/PR (incident-like)",
                        occurred_at=occurred,
                        discovered_at=now,
                        protocol=None,
                        chain=None,
                        amount_usd=None,
                        # Generic label; keep semantics for downstream filters/analytics.
                        category="github_signal",
                        description=None,
                        tags=tags,
                        raw=it,
                    )
                )

        self.logger.info(f"GitHub incident-like items collected: {len(incidents)} (since {since_days}d)")
        return incidents


class PublicIncidentsDatasetBuilder:
    def __init__(self) -> None:
        logger = logging.getLogger("incidents_public_dataset")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger
        self.http = HttpClient()

    def collect_incidents(
        self,
        *,
        since_days: int = 365,
        write_raw: bool = True,
        include_defillama: bool = True,
        include_eigenlayer_blog: bool = True,
        include_eigenlayer_sitemap: bool = True,
        include_rekt: bool = True,
        include_github: bool = True,
        repos: Optional[List[str]] = None,
    ) -> Tuple[List[Incident], List[str]]:
        """
        Collect raw incident-like objects from configured sources (no filtering, no DataFrame).
        Returns (incidents, ecosystem_terms).
        """
        now = _utc_now()
        since_dt = now - timedelta(days=max(0, int(since_days)))
        ecosystem_terms = _load_ecosystem_terms(self.logger)

        incidents: List[Incident] = []

        def safe_collect(name: str, fn) -> None:
            try:
                incidents.extend(fn())
            except Exception as e:
                self.logger.warning(f"Collector failed ({name}): {type(e).__name__}: {e}")

        if include_defillama:
            safe_collect(
                "defillama_hacks",
                lambda: DeFiLlamaHacksCollector(self.http, self.logger).collect(since_dt=since_dt, write_raw=write_raw),
            )
        if include_eigenlayer_blog:
            safe_collect(
                "eigenlayer_blog_feed",
                lambda: EigenLayerBlogCollector(self.http, self.logger).collect(since_dt=since_dt, write_raw=write_raw),
            )
        if include_eigenlayer_sitemap:
            safe_collect(
                "eigenlayer_sitemap",
                lambda: EigenLayerSitemapCollector(self.http, self.logger).collect(since_dt=since_dt, write_raw=write_raw),
            )
        if include_rekt:
            safe_collect(
                "rekt",
                lambda: RektSitemapCollector(self.http, self.logger).collect(
                    since_dt=since_dt, write_raw=write_raw, ecosystem_terms=ecosystem_terms
                ),
            )
        if include_github:
            safe_collect(
                "github_search",
                lambda: GitHubSignalsCollector(self.logger).collect(since_days=since_days, repos=repos, write_raw=write_raw),
            )

        return incidents, ecosystem_terms

    def incidents_to_df(self, incidents: List[Incident]) -> pd.DataFrame:
        df = pd.DataFrame([i.to_row() for i in incidents])
        if df.empty:
            return df
        df = df.drop_duplicates(subset=["incident_id"]).copy()
        df["_occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
        df = df.sort_values(by=["_occurred_at", "source"], ascending=[False, True]).drop(columns=["_occurred_at"])
        return df

    def filter_restaking(self, incidents: List[Incident], ecosystem_terms: List[str]) -> pd.DataFrame:
        kept_rows: List[Dict[str, Any]] = []
        for inc in incidents:
            if not _is_restaking_related(inc, ecosystem_terms):
                continue
            # GitHub is particularly noisy: keep only explicitly incident/security-related items.
            if inc.source == "github_search" and not _is_incident_like(inc):
                continue
            kept_rows.append(inc.to_row())
        df = pd.DataFrame(kept_rows)
        if df.empty:
            return df
        df = df.drop_duplicates(subset=["incident_id"]).copy()
        df["_occurred_at"] = pd.to_datetime(df["occurred_at"], utc=True, errors="coerce")
        df = df.sort_values(by=["_occurred_at", "source"], ascending=[False, True]).drop(columns=["_occurred_at"])
        return df

    def build(
        self,
        *,
        since_days: int = 365,
        write_raw: bool = True,
        include_defillama: bool = True,
        include_eigenlayer_blog: bool = True,
        include_eigenlayer_sitemap: bool = True,
        include_rekt: bool = True,
        include_github: bool = True,
        repos: Optional[List[str]] = None,
        restaking_only: bool = True,
    ) -> pd.DataFrame:
        incidents, ecosystem_terms = self.collect_incidents(
            since_days=since_days,
            write_raw=write_raw,
            include_defillama=include_defillama,
            include_eigenlayer_blog=include_eigenlayer_blog,
            include_eigenlayer_sitemap=include_eigenlayer_sitemap,
            include_rekt=include_rekt,
            include_github=include_github,
            repos=repos,
        )
        if restaking_only:
            return self.filter_restaking(incidents, ecosystem_terms)
        return self.incidents_to_df(incidents)

    def write(self, df: pd.DataFrame, *, out_csv: Path) -> Path:
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_csv, index=False)
        self.logger.info(f"Wrote incidents dataset: {out_csv} (rows={len(df)})")
        return out_csv

    def write_with_backup(
        self,
        df: pd.DataFrame,
        *,
        out_csv: Path,
        backup_suffix: str = "_prev",
    ) -> Tuple[Optional[Path], Path]:
        """
        If out_csv already exists, move it aside as a backup (best-effort) before overwriting.
        Returns (backup_path, out_csv_path).
        """
        out_csv = Path(out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        backup_path: Optional[Path] = None
        try:
            if out_csv.exists():
                backup_path = out_csv.with_name(out_csv.stem + f"{backup_suffix}" + out_csv.suffix)
                # Keep the latest backup only (overwrite).
                out_csv.replace(backup_path)
                self.logger.info(f"Backed up previous dataset to: {backup_path}")
        except Exception as e:
            self.logger.warning(f"Failed to backup existing dataset: {type(e).__name__}: {e}")

        df.to_csv(out_csv, index=False)
        self.logger.info(f"Wrote incidents dataset: {out_csv} (rows={len(df)})")
        return backup_path, out_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Build best-effort public incidents dataset (real sources).")
    parser.add_argument("--since-days", type=int, default=365, help="How far back to collect.")
    parser.add_argument(
        "--out-csv",
        type=str,
        default=str(DERIVED_DIR / "incidents_public.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--out-csv-all",
        type=str,
        default="",
        help="Optional output path for the unfiltered dataset. If omitted, uses '<out-csv stem>_all.csv'.",
    )
    parser.add_argument("--no-raw", action="store_true", help="Do not write raw snapshots under data/derived/incidents_raw/")
    parser.add_argument("--no-defillama", action="store_true", help="Disable DeFiLlama hacks collector.")
    parser.add_argument("--no-eigenlayer-blog", action="store_true", help="Disable EigenLayer blog collector.")
    parser.add_argument("--no-eigenlayer-sitemap", action="store_true", help="Disable EigenLayer sitemap collector.")
    parser.add_argument("--no-rekt", action="store_true", help="Disable REKT sitemap collector.")
    parser.add_argument("--no-github", action="store_true", help="Disable GitHub signals collector.")
    parser.add_argument(
        "--no-restaking-only",
        action="store_true",
        help="Disable EigenLayer/restaking relevance filtering (keeps general incidents too).",
    )
    parser.add_argument(
        "--repos",
        type=str,
        default="",
        help="Optional comma-separated owner/repo list for GitHub collection. If omitted, auto-discovers.",
    )
    args = parser.parse_args()

    repos = [r.strip() for r in str(args.repos).split(",") if r.strip()] if str(args.repos).strip() else None

    builder = PublicIncidentsDatasetBuilder()
    incidents, ecosystem_terms = builder.collect_incidents(
        since_days=int(args.since_days),
        write_raw=not bool(args.no_raw),
        include_defillama=not bool(args.no_defillama),
        include_eigenlayer_blog=not bool(args.no_eigenlayer_blog),
        include_eigenlayer_sitemap=not bool(args.no_eigenlayer_sitemap),
        include_rekt=not bool(args.no_rekt),
        include_github=not bool(args.no_github),
        repos=repos,
    )

    df_all = builder.incidents_to_df(incidents)
    out_csv = Path(args.out_csv)
    out_csv_all = Path(args.out_csv_all) if str(args.out_csv_all).strip() else out_csv.with_name(out_csv.stem + "_all" + out_csv.suffix)
    builder.write(df_all, out_csv=out_csv_all)

    if bool(args.no_restaking_only):
        df_filtered = df_all
    else:
        df_filtered = builder.filter_restaking(incidents, ecosystem_terms)
    builder.write_with_backup(df_filtered, out_csv=out_csv, backup_suffix="_prev")


if __name__ == "__main__":
    # Ensure caches are writable even in restricted environments
    os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))
    Path(os.environ["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)
    main()


