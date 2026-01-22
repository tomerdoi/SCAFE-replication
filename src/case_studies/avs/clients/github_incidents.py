"""
GitHub incidents/case-study mining utility (FREE API with rate limits).

What it collects (per repo):
- "Incident-like" issues/PRs via GitHub Search API (security, incident, outage, postmortem, CVE/GHSA keywords)
- Releases (useful for upgrade cadence / operational churn case studies)

Outputs JSON under data/ for later analysis / labeling / case studies.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.config.global_conf import CONF
from src.paths import AVS_DATA_DIR


DATA_DIR = AVS_DATA_DIR


class GitHubError(RuntimeError):
    pass


@dataclass(slots=True)
class GitHubClient:
    base_url: str = CONF.GITHUB_API_BASE_URL
    token: Optional[str] = CONF.GITHUB_TOKEN
    timeout_secs: int = CONF.HTTP_TIMEOUT_SECS
    max_retries: int = CONF.HTTP_MAX_RETRIES
    backoff_secs: float = CONF.HTTP_BACKOFF_SECS
    _session: requests.Session = field(init=False, repr=False)

    def __post_init__(self) -> None:
        s = requests.Session()
        retry = Retry(
            total=max(0, int(self.max_retries)),
            backoff_factor=float(self.backoff_secs),
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=("GET",),
            raise_on_status=False,
        )
        s.mount("https://", HTTPAdapter(max_retries=retry))
        s.mount("http://", HTTPAdapter(max_retries=retry))
        self._session = s

    def _headers(self) -> Dict[str, str]:
        h = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "sc_review_risk_score_pipeline",
        }
        if self.token:
            # Works for classic + fine-grained tokens (Bearer is accepted).
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def get_json(self, path: str, params: Optional[Dict[str, Any]] = None) -> Tuple[Any, requests.Response]:
        url = f"{self.base_url.rstrip('/')}/{path.lstrip('/')}"
        resp = self._session.get(url, headers=self._headers(), params=params, timeout=self.timeout_secs)
        try:
            data = resp.json()
        except Exception:
            data = resp.text
        return data, resp


class GitHubIncidentsUtil:
    def __init__(self) -> None:
        logger = logging.getLogger("github_incidents_util")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
            logger.addHandler(h)
        self.logger = logger
        self.client = GitHubClient()
        if not self.client.token:
            self.logger.warning(
                "No GITHUB_TOKEN configured. This will work, but GitHub search rate limits are very small; "
                "for many repos you may hit rate limits."
            )

    def ensure_dirs(self) -> None:
        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during ensure_dirs", exc_info=True)
            raise

    def discover_repos_from_u1_metadata(self, u1_latest_avs_balances_path: Optional[Path] = None) -> List[str]:
        """
        Best-effort discovery of GitHub repos from existing u--1 AVS metadata.

        Note: u--1 metadata often includes logo URLs on raw.githubusercontent.com which embed owner/repo.
        We extract those as a starting list of likely AVS-related repos.
        """
        try:
            if u1_latest_avs_balances_path is None:
                u1_latest_avs_balances_path = DATA_DIR / "u1_latest_avs_balances.json"
            if not u1_latest_avs_balances_path.exists():
                return []

            text = u1_latest_avs_balances_path.read_text(encoding="utf-8", errors="ignore")
            repos: set[str] = set()

            # raw.githubusercontent.com/<owner>/<repo>/...
            for m in re.finditer(
                r"https?://raw\.githubusercontent\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)/", text
            ):
                owner, repo = m.group(1), m.group(2)
                if owner and repo:
                    repos.add(f"{owner}/{repo}")

            # github.com/<owner>/<repo>(/...)
            for m in re.finditer(r"https?://github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", text):
                owner, repo = m.group(1), m.group(2)
                if owner and repo:
                    repos.add(f"{owner}/{repo}")

            return sorted(repos)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during discover_repos_from_u1_metadata", exc_info=True)
            raise

    def _iso_date_days_ago(self, days: int) -> str:
        try:
            d = datetime.now(timezone.utc) - timedelta(days=max(0, int(days)))
            return d.date().isoformat()
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _iso_date_days_ago", exc_info=True)
            raise

    def _handle_rate_limit(self, resp: requests.Response, *, auto_sleep: bool) -> None:
        """
        If we hit a rate limit, optionally sleep until reset.
        """
        try:
            remaining = resp.headers.get("X-RateLimit-Remaining")
            reset = resp.headers.get("X-RateLimit-Reset")
            if remaining == "0" and reset and auto_sleep:
                try:
                    reset_ts = int(reset)
                    now_ts = int(time.time())
                    sleep_s = max(1, reset_ts - now_ts + 1)
                except Exception:
                    sleep_s = 60
                self.logger.warning(f"Rate limited by GitHub; sleeping {sleep_s}s until reset")
                time.sleep(sleep_s)
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during _handle_rate_limit", exc_info=True)
            raise

    def search_incident_issues(
        self,
        repo_full_name: str,
        since_days: int = 365,
        *,
        per_page: int = 100,
        max_pages: int = 10,
        auto_sleep_on_rate_limit: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Uses GitHub Search API to find issues/PRs that look like incidents / postmortems / security events.
        """
        try:
            since = self._iso_date_days_ago(since_days)
            # GitHub Search API has a hard limit on boolean operators per query.
            # Keep to 5 ORs (i.e., 6 terms is too many).
            keywords = "incident OR postmortem OR outage OR exploit OR vulnerability"
            q = f"repo:{repo_full_name} ({keywords}) updated:>={since}"

            items: List[Dict[str, Any]] = []
            for page in range(1, max(1, int(max_pages)) + 1):
                data, resp = self.client.get_json(
                    "/search/issues",
                    params={"q": q, "sort": "updated", "order": "desc", "per_page": max(1, int(per_page)), "page": page},
                )
                if resp.status_code == 403:
                    self._handle_rate_limit(resp, auto_sleep=auto_sleep_on_rate_limit)
                if resp.status_code >= 400:
                    raise GitHubError(f"GitHub search failed: HTTP {resp.status_code} | {data}")

                page_items = data.get("items", []) if isinstance(data, dict) else []
                if not isinstance(page_items, list):
                    break
                if not page_items:
                    break
                items.extend(page_items)

                # stop early if less than requested returned (no more pages)
                if len(page_items) < max(1, int(per_page)):
                    break

            # keep only high-signal fields
            out: List[Dict[str, Any]] = []
            for it in items:
                if not isinstance(it, dict):
                    continue
                out.append(
                    {
                        "number": it.get("number"),
                        "title": it.get("title"),
                        "state": it.get("state"),
                        "html_url": it.get("html_url"),
                        "created_at": it.get("created_at"),
                        "updated_at": it.get("updated_at"),
                        "closed_at": it.get("closed_at"),
                        "labels": [l.get("name") for l in (it.get("labels") or []) if isinstance(l, dict)],
                        "is_pull_request": "pull_request" in it,
                    }
                )
            return out
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during search_incident_issues", exc_info=True)
            raise

    def list_releases(
        self,
        repo_full_name: str,
        since_days: int = 365,
        *,
        per_page: int = 100,
        max_pages: int = 10,
        auto_sleep_on_rate_limit: bool = True,
    ) -> List[Dict[str, Any]]:
        try:
            since_dt = datetime.now(timezone.utc) - timedelta(days=max(0, int(since_days)))
            releases: List[Dict[str, Any]] = []
            owner, repo = repo_full_name.split("/", 1)

            for page in range(1, max(1, int(max_pages)) + 1):
                data, resp = self.client.get_json(
                    f"/repos/{owner}/{repo}/releases",
                    params={"per_page": max(1, int(per_page)), "page": page},
                )
                if resp.status_code == 403:
                    self._handle_rate_limit(resp, auto_sleep=auto_sleep_on_rate_limit)
                if resp.status_code >= 400:
                    raise GitHubError(f"GitHub releases failed: HTTP {resp.status_code} | {data}")

                if not isinstance(data, list):
                    break
                if not data:
                    break

                for r in data:
                    if not isinstance(r, dict):
                        continue
                    published_at = r.get("published_at")
                    try:
                        pub_dt = (
                            datetime.fromisoformat(str(published_at).replace("Z", "+00:00"))
                            if published_at
                            else None
                        )
                    except Exception:
                        pub_dt = None
                    if pub_dt and pub_dt < since_dt:
                        continue
                    releases.append(
                        {
                            "id": r.get("id"),
                            "tag_name": r.get("tag_name"),
                            "name": r.get("name"),
                            "draft": r.get("draft"),
                            "prerelease": r.get("prerelease"),
                            "published_at": r.get("published_at"),
                            "html_url": r.get("html_url"),
                        }
                    )

                if len(data) < max(1, int(per_page)):
                    break

            return releases
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during list_releases", exc_info=True)
            raise

    def fetch_repo_case_study_bundle(
        self, repo_full_name: str, since_days: int = 365, *, auto_sleep_on_rate_limit: bool = True
    ) -> Dict[str, Any]:
        try:
            issues = self.search_incident_issues(
                repo_full_name, since_days=since_days, auto_sleep_on_rate_limit=auto_sleep_on_rate_limit
            )
            releases = self.list_releases(
                repo_full_name, since_days=since_days, auto_sleep_on_rate_limit=auto_sleep_on_rate_limit
            )
            return {
                "repo": repo_full_name,
                "since_days": since_days,
                "incident_issues_prs": issues,
                "releases": releases,
            }
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during fetch_repo_case_study_bundle", exc_info=True)
            raise

    def run(self, repos: Sequence[str], since_days: int = 365, *, out_path: Optional[Path] = None) -> Path:
        try:
            self.ensure_dirs()
            repos = [r.strip() for r in repos if isinstance(r, str) and r.strip()]
            if not repos:
                raise ValueError("No repos provided. Use owner/repo strings.")

            bundles = []
            errors: List[Dict[str, Any]] = []
            for r in repos:
                self.logger.info(f"Fetching GitHub case-study signals for {r}")
                try:
                    bundles.append(self.fetch_repo_case_study_bundle(r, since_days=since_days))
                except Exception as e:
                    # Keep going: repo might be missing/private/renamed, or rate-limited.
                    self.logger.error(f"Exception {e} occurred while fetching {r}", exc_info=True)
                    errors.append({"repo": r, "error": str(e)})

            payload = {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "since_days": since_days,
                "repos": bundles,
                "errors": errors,
            }

            if out_path is None:
                out_path = DATA_DIR / "github_case_studies.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self.logger.info(f"Wrote: {out_path}")
            return out_path
        except Exception as e:
            self.logger.error(f"Exception {e} occurred during run", exc_info=True)
            raise


if __name__ == "__main__":
    # Edit these locally (kept intentionally simple; no argparse).
    SINCE_DAYS = 365
    # Base repos to scan for EigenLayer restaking risk case studies (core + operator tooling).
    BASE_REPOS = [
        # EigenLayer / Layr Labs core + AVS code
        "Layr-Labs/eigenlayer-contracts",
        "Layr-Labs/eigensdk-go",
        "Layr-Labs/eigenda",
        # Common dependencies frequently used by AVSs (security/event context)
        "OpenZeppelin/openzeppelin-contracts",
    ]

    util = GitHubIncidentsUtil()
    # Auto-discover additional AVS-related repos from your existing u--1 metadata file (logo/links).
    DISCOVERED = util.discover_repos_from_u1_metadata()
    REPOS = sorted(set(BASE_REPOS + DISCOVERED))

    util.run(REPOS, since_days=int(SINCE_DAYS))

