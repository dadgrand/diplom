from __future__ import annotations

import argparse
import hashlib
import re
import time
import urllib3
from collections import deque
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from lxml import html

from risk_pipeline.financial_reports import _detect_report_extension


YEARS = {"2021", "2022", "2023", "2024", "2025"}
DOC_EXTENSIONS = {".pdf", ".xlsx", ".xlsm", ".zip", ".xls", ".doc", ".docx"}
REPORT_KEYWORDS = [
    "ifrs",
    "msfo",
    "мсфо",
    "financial",
    "финансов",
    "statement",
    "отчет",
    "отчёт",
    "annual",
    "годов",
    "interim",
    "quarter",
    "results",
    "результат",
    "consolidated",
    "консолид",
]
PAGE_KEYWORDS = [
    "report",
    "reports",
    "financial",
    "results",
    "annual",
    "statement",
    "investor",
    "investors",
    "disclosure",
    "отчет",
    "отчеты",
    "отчёты",
    "финансов",
    "результат",
    "инвест",
    "раскрыт",
]


@dataclass
class LinkCandidate:
    ticker: str
    company_name: str
    source_page: str
    url: str
    text: str
    depth: int
    score: int


def normalize_text(value: str | None) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def clean_filename(value: str, max_len: int = 120) -> str:
    value = re.sub(r"[^\w.\-]+", "_", value, flags=re.UNICODE).strip("._")
    return value[:max_len] or "report"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def same_site(url: str, root_url: str) -> bool:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    root_host = urlparse(root_url).netloc.lower().removeprefix("www.")
    return host == root_host or host.endswith("." + root_host)


def canonical_doc_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.query in {"dl=1", "download=1"}:
        return parsed._replace(query="").geturl()
    return url


def extension_from_url(url: str) -> str:
    return Path(urlparse(url).path).suffix.lower()


def has_year(text: str) -> bool:
    return any(year in text for year in YEARS)


def infer_report_type(text: str) -> str:
    lower = text.lower()
    if "annual" in lower or "годов" in lower:
        return "annual_report"
    if "9m" in lower or "9 months" in lower or "9 мес" in lower:
        return "ifrs_9m"
    if "h1" in lower or "6m" in lower or "half" in lower or "полуг" in lower:
        return "ifrs_h1"
    if "q1" in lower or "1q" in lower or "quarter" in lower or "кварт" in lower:
        return "ifrs_q"
    if "ifrs" in lower or "мсфо" in lower or "consolidated" in lower:
        return "ifrs"
    return "report"


def infer_period_end(text: str) -> str | None:
    lower = text.lower()
    years = sorted({int(y) for y in re.findall(r"\b(202[1-5])\b", lower)})
    if not years:
        return None
    year = years[-1]
    if re.search(r"\b(q1|1q|3\s*months|3\s*месяц|1\s*кв)", lower):
        return f"{year}-03-31"
    if re.search(r"\b(q2|2q|h1|1h|6m|half|6\s*months|6\s*месяц|полуг|2\s*кв)", lower):
        return f"{year}-06-30"
    if re.search(r"\b(q3|3q|9m|9\s*months|9\s*месяц|3\s*кв)", lower):
        return f"{year}-09-30"
    return f"{year}-12-31"


def parse_last_modified(value: str | None) -> str | None:
    if not value:
        return None
    try:
        dt = parsedate_to_datetime(value)
    except Exception:
        return None
    if dt is None:
        return None
    date = dt.date().isoformat()
    return date if "2022-01-01" <= date <= "2025-08-31" else None


def score_link(url: str, text: str) -> int:
    blob = f"{url} {text}".lower()
    score = 0
    if extension_from_url(url) in DOC_EXTENSIONS:
        score += 5
    if has_year(blob):
        score += 4
    score += sum(2 for keyword in REPORT_KEYWORDS if keyword in blob)
    if "sustainability" in blob or "esg" in blob or "устойчив" in blob:
        score -= 6
    if "presentation" in blob or "презентац" in blob:
        score -= 2
    return score


def page_link_score(url: str, text: str) -> int:
    blob = f"{url} {text}".lower()
    return sum(1 for keyword in PAGE_KEYWORDS if keyword in blob) + (2 if has_year(blob) else 0)


def fetch(session: requests.Session, url: str, *, timeout: int = 35) -> requests.Response:
    response = session.get(url, timeout=timeout, allow_redirects=True)
    response.raise_for_status()
    return response


def extract_links(page_url: str, content: bytes) -> list[tuple[str, str]]:
    raw = content.decode("utf-8", errors="ignore")
    raw_for_urls = raw.replace("\\/", "/")
    try:
        doc = html.fromstring(content)
    except Exception:
        doc = None
    links: list[tuple[str, str]] = []
    if doc is not None:
        doc.make_links_absolute(page_url)
        for node in doc.xpath("//a[@href]"):
            href = node.get("href")
            text = normalize_text(" ".join([node.get("title") or "", node.get("aria-label") or "", " ".join(node.itertext())]))
            if href:
                links.append((href, text))
    # Many issuer pages keep report links in inline JSON/JS rather than anchors.
    # Keep nearby stripped text as evidence for scoring and period inference.
    raw_url_re = r"https?://[^\s\"'<>]+?\.(?:pdf|xlsx|xlsm|zip|xls|docx?)(?:\?[^\s\"'<>]*)?"
    for match in re.finditer(raw_url_re, raw_for_urls, flags=re.IGNORECASE):
        url = match.group(0).replace("\\/", "/")
        start = max(0, match.start() - 900)
        end = min(len(raw_for_urls), match.end() + 300)
        context = normalize_text(re.sub(r"<[^>]+>", " ", raw_for_urls[start:end]))
        links.append((urljoin(page_url, url), context))
    return links


def discover_candidates_for_source(
    session: requests.Session,
    *,
    ticker: str,
    company_name: str,
    root_url: str,
    max_pages: int,
    max_depth: int,
    sleep: float,
) -> tuple[list[LinkCandidate], list[dict[str, object]]]:
    queue = deque([(root_url, 0)])
    visited: set[str] = set()
    candidates: list[LinkCandidate] = []
    page_rows: list[dict[str, object]] = []

    while queue and len(visited) < max_pages:
        page_url, depth = queue.popleft()
        if page_url in visited or depth > max_depth:
            continue
        visited.add(page_url)
        try:
            response = fetch(session, page_url)
            content_type = response.headers.get("Content-Type", "")
            page_rows.append(
                {
                    "ticker": ticker,
                    "source_page": page_url,
                    "depth": depth,
                    "status": f"ok:{response.status_code}",
                    "content_type": content_type,
                    "bytes": len(response.content),
                    "error": None,
                }
            )
        except Exception as exc:
            page_rows.append(
                {
                    "ticker": ticker,
                    "source_page": page_url,
                    "depth": depth,
                    "status": "failed",
                    "content_type": None,
                    "bytes": 0,
                    "error": str(exc)[:500],
                }
            )
            continue

        if "html" not in content_type.lower() and extension_from_url(page_url) in DOC_EXTENSIONS:
            candidates.append(LinkCandidate(ticker, company_name, page_url, page_url, "", depth, score_link(page_url, "")))
            continue

        for href, text in extract_links(response.url, response.content):
            parsed = urlparse(href)
            if parsed.scheme not in {"http", "https"}:
                continue
            link_score = score_link(href, text)
            if link_score >= 7:
                candidates.append(LinkCandidate(ticker, company_name, response.url, href, text, depth, link_score))
            if depth < max_depth and same_site(href, root_url) and extension_from_url(href) not in DOC_EXTENSIONS:
                if page_link_score(href, text) >= 1 and href not in visited:
                    queue.append((href, depth + 1))
        time.sleep(sleep)

    unique: dict[str, LinkCandidate] = {}
    for candidate in sorted(candidates, key=lambda c: c.score, reverse=True):
        unique.setdefault(canonical_doc_url(candidate.url), candidate)
    return list(unique.values()), page_rows


def download_candidate(
    session: requests.Session,
    candidate: LinkCandidate,
    *,
    reports_dir: Path,
    sleep: float,
) -> dict[str, object]:
    combined = normalize_text(f"{candidate.text} {candidate.url}")
    out_dir = reports_dir / candidate.ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        response = fetch(session, candidate.url, timeout=60)
        content = response.content
        detected_ext = _detect_report_extension(
            url=response.url,
            content_type=response.headers.get("Content-Type"),
            content_disposition=response.headers.get("Content-Disposition"),
            content=content,
        )
        if detected_ext in {".bin", ".html", ".htm"} and extension_from_url(response.url) not in DOC_EXTENSIONS:
            raise ValueError(f"not a report document: detected_ext={detected_ext}")
        digest = sha256_bytes(content)
        period_end = infer_period_end(combined)
        publish_date = parse_last_modified(response.headers.get("Last-Modified"))
        report_type = infer_report_type(combined)
        year_part = (period_end or "unknown").replace("-", "")
        base_name = clean_filename(f"{candidate.ticker}_{year_part}_{publish_date or 'unknown'}_{report_type}_{digest[:10]}{detected_ext}")
        path = out_dir / base_name
        existing = next(out_dir.glob(f"*_{digest[:10]}{detected_ext}"), None)
        if existing is not None:
            path = existing
            base_name = existing.name
        if not path.exists():
            path.write_bytes(content)
        time.sleep(sleep)
        return {
            "ticker": candidate.ticker,
            "company_name": candidate.company_name,
            "report_period_end": period_end,
            "publish_date": publish_date,
            "report_type": report_type,
            "accounting_standard": "IFRS" if "ifrs" in report_type or "ifrs" in combined.lower() or "мсфо" in combined.lower() else None,
            "language": None,
            "source_name": "issuer_ir_auto",
            "source_page": candidate.source_page,
            "source_url": response.url,
            "local_path": str(Path(candidate.ticker) / base_name),
            "download_status": f"downloaded:{response.status_code}",
            "content_type": response.headers.get("Content-Type"),
            "detected_extension": detected_ext,
            "file_sha256": digest,
            "bytes": len(content),
            "link_text": candidate.text,
            "score": candidate.score,
            "error": None,
        }
    except Exception as exc:
        return {
            "ticker": candidate.ticker,
            "company_name": candidate.company_name,
            "report_period_end": infer_period_end(combined),
            "publish_date": None,
            "report_type": infer_report_type(combined),
            "accounting_standard": None,
            "language": None,
            "source_name": "issuer_ir_auto",
            "source_page": candidate.source_page,
            "source_url": candidate.url,
            "local_path": None,
            "download_status": "failed",
            "content_type": None,
            "detected_extension": extension_from_url(candidate.url),
            "file_sha256": None,
            "bytes": 0,
            "link_text": candidate.text,
            "score": candidate.score,
            "error": str(exc)[:500],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Best-effort issuer IR report downloader")
    parser.add_argument("--sources", default="data/report_sources_ru_bluechips.csv")
    parser.add_argument("--reports-dir", default="data/raw/reports")
    parser.add_argument("--registry-output", default="data/raw/report_registry_auto.csv")
    parser.add_argument("--links-output", default="data/raw/report_links_auto.csv")
    parser.add_argument("--pages-output", default="data/raw/report_pages_auto.csv")
    parser.add_argument("--max-pages-per-issuer", type=int, default=14)
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-downloads-per-issuer", type=int, default=18)
    parser.add_argument("--sleep", type=float, default=0.2)
    parser.add_argument("--insecure", action="store_true", help="disable TLS verification for issuers with broken certificate chains")
    args = parser.parse_args()

    sources = pd.read_csv(args.sources)
    session = requests.Session()
    if args.insecure:
        session.verify = False
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    session.headers.update(
        {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) risk-pipeline-academic-report-collector/0.1",
            "Accept": "text/html,application/xhtml+xml,application/pdf,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet,*/*;q=0.8",
        }
    )

    reports_dir = Path(args.reports_dir)
    registry_rows: list[dict[str, object]] = []
    link_rows: list[dict[str, object]] = []
    page_rows_all: list[dict[str, object]] = []

    for _, row in sources.iterrows():
        ticker = str(row["ticker"]).upper().strip()
        issuer_url = row.get("issuer_url")
        if not isinstance(issuer_url, str) or not issuer_url.strip():
            continue
        company_name = str(row.get("company_name") or ticker)
        print(f"[{ticker}] crawling {issuer_url}", flush=True)
        candidates, page_rows = discover_candidates_for_source(
            session,
            ticker=ticker,
            company_name=company_name,
            root_url=issuer_url,
            max_pages=args.max_pages_per_issuer,
            max_depth=args.max_depth,
            sleep=args.sleep,
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
            registry_rows.append(download_candidate(session, candidate, reports_dir=reports_dir, sleep=args.sleep))

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


if __name__ == "__main__":
    main()
