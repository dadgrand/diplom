from __future__ import annotations

import argparse
import asyncio
from collections import deque
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
from crawl4ai import AsyncWebCrawler, BrowserConfig, CacheMode, CrawlerRunConfig

from download_reports_auto import (
    LinkCandidate,
    canonical_doc_url,
    download_candidate,
    extension_from_url,
    extract_links,
    page_link_score,
    same_site,
    score_link,
)


def normalize_url(url: str) -> str:
    return url.split("#", 1)[0]


def rows_from_crawl_links(result, page_url: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for bucket in ["internal", "external"]:
        for item in (getattr(result, "links", None) or {}).get(bucket, []):
            href = item.get("href")
            if not href:
                continue
            text = " ".join(str(item.get(k) or "") for k in ["text", "title"]).strip()
            rows.append((urljoin(page_url, href), text))
    return rows


async def crawl_candidates_for_source(
    crawler: AsyncWebCrawler,
    *,
    ticker: str,
    company_name: str,
    root_url: str,
    max_pages: int,
    max_depth: int,
    run_config: CrawlerRunConfig,
) -> tuple[list[LinkCandidate], list[dict[str, object]]]:
    queue = deque([(root_url, 0)])
    visited: set[str] = set()
    candidates: list[LinkCandidate] = []
    page_rows: list[dict[str, object]] = []

    while queue and len(visited) < max_pages:
        page_url, depth = queue.popleft()
        page_url = normalize_url(page_url)
        if page_url in visited or depth > max_depth:
            continue
        visited.add(page_url)
        try:
            result = await crawler.arun(url=page_url, config=run_config)
            html = (getattr(result, "html", None) or "").replace("\\/", "/")
            status_code = getattr(result, "status_code", None)
            success = bool(getattr(result, "success", False))
            error_message = getattr(result, "error_message", None)
            page_rows.append(
                {
                    "ticker": ticker,
                    "source_page": page_url,
                    "depth": depth,
                    "status": f"ok:{status_code}" if success else "failed",
                    "status_code": status_code,
                    "html_bytes": len(html.encode("utf-8", errors="ignore")),
                    "error": error_message,
                }
            )
            if not html and not success:
                continue
        except Exception as exc:
            page_rows.append(
                {
                    "ticker": ticker,
                    "source_page": page_url,
                    "depth": depth,
                    "status": "failed",
                    "status_code": None,
                    "html_bytes": 0,
                    "error": str(exc)[:500],
                }
            )
            continue

        discovered_links = []
        discovered_links.extend(rows_from_crawl_links(result, page_url))
        discovered_links.extend(extract_links(page_url, html.encode("utf-8", errors="ignore")))

        for href, text in discovered_links:
            href = normalize_url(href)
            parsed = urlparse(href)
            if parsed.scheme not in {"http", "https"}:
                continue
            link_score = score_link(href, text)
            if link_score >= 7:
                candidates.append(LinkCandidate(ticker, company_name, page_url, href, text, depth, link_score))
            if depth < max_depth and same_site(href, root_url) and extension_from_url(href) == "":
                if page_link_score(href, text) >= 1 and href not in visited:
                    queue.append((href, depth + 1))

    unique: dict[str, LinkCandidate] = {}
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        unique.setdefault(canonical_doc_url(candidate.url), candidate)
    return list(unique.values()), page_rows


async def main_async(args: argparse.Namespace) -> None:
    sources = pd.read_csv(args.sources)
    browser_config = BrowserConfig(
        headless=True,
        ignore_https_errors=True,
        enable_stealth=True,
        verbose=False,
        viewport_width=1365,
        viewport_height=900,
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    )
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS,
        page_timeout=args.page_timeout,
        wait_until="domcontentloaded",
        delay_before_return_html=args.delay,
        scan_full_page=True,
        max_scroll_steps=args.max_scroll_steps,
        simulate_user=True,
        override_navigator=True,
        magic=True,
        remove_overlay_elements=True,
        remove_consent_popups=True,
        verbose=False,
    )

    registry_rows: list[dict[str, object]] = []
    link_rows: list[dict[str, object]] = []
    page_rows_all: list[dict[str, object]] = []
    async with AsyncWebCrawler(config=browser_config) as crawler:
        for _, row in sources.iterrows():
            ticker = str(row["ticker"]).upper().strip()
            issuer_url = str(row["issuer_url"]).strip()
            company_name = str(row.get("company_name") or ticker)
            print(f"[{ticker}] crawl4ai {issuer_url}", flush=True)
            candidates, page_rows = await crawl_candidates_for_source(
                crawler,
                ticker=ticker,
                company_name=company_name,
                root_url=issuer_url,
                max_pages=args.max_pages_per_issuer,
                max_depth=args.max_depth,
                run_config=run_config,
            )
            page_rows_all.extend(page_rows)
            candidates = sorted(candidates, key=lambda c: c.score, reverse=True)
            for candidate in candidates:
                link_rows.append(
                    {
                        "ticker": candidate.ticker,
                        "company_name": candidate.company_name,
                        "source_page": candidate.source_page,
                        "source_url": candidate.url,
                        "link_text": candidate.text,
                        "depth": candidate.depth,
                        "score": candidate.score,
                    }
                )
            for candidate in candidates[: args.max_downloads_per_issuer]:
                registry_rows.append(
                    download_candidate(
                        requests_session,
                        candidate,
                        reports_dir=Path(args.reports_dir),
                        sleep=args.sleep,
                    )
                )

    for path, rows in [
        (args.registry_output, registry_rows),
        (args.links_output, link_rows),
        (args.pages_output, page_rows_all),
    ]:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(out, index=False)

    registry = pd.DataFrame(registry_rows)
    if registry.empty:
        print("No documents downloaded")
    else:
        print(registry.groupby("ticker")["download_status"].value_counts(dropna=False).to_string())
        print(f"saved registry: {args.registry_output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Crawl4AI-based issuer IR report downloader")
    parser.add_argument("--sources", required=True)
    parser.add_argument("--reports-dir", default="data/raw/reports")
    parser.add_argument("--registry-output", default="data/raw/report_registry_crawl4ai.csv")
    parser.add_argument("--links-output", default="data/raw/report_links_crawl4ai.csv")
    parser.add_argument("--pages-output", default="data/raw/report_pages_crawl4ai.csv")
    parser.add_argument("--max-pages-per-issuer", type=int, default=10)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-downloads-per-issuer", type=int, default=18)
    parser.add_argument("--max-scroll-steps", type=int, default=5)
    parser.add_argument("--page-timeout", type=int, default=70000)
    parser.add_argument("--delay", type=float, default=4.0)
    parser.add_argument("--sleep", type=float, default=0.12)
    args = parser.parse_args()

    global requests_session
    import requests
    import urllib3

    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    requests_session = requests.Session()
    requests_session.verify = False
    requests_session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
