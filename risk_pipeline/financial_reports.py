from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urljoin, urlparse

import numpy as np
import pandas as pd
import requests

from .data_sources import DownloadError, load_local_csv, save_frame

LOGGER = logging.getLogger(__name__)

E_DISCLOSURE_BASE = "https://www.e-disclosure.ru"
E_DISCLOSURE_FILES_URL = f"{E_DISCLOSURE_BASE}/portal/files.aspx"
E_DISCLOSURE_FILELOAD_URL = f"{E_DISCLOSURE_BASE}/portal/FileLoad.ashx"

REPORT_DATE_COLUMNS = ("report_period_end", "publish_date", "downloaded_at")
NUMERIC_EPS = 1e-12
SUPPORTED_REPORT_EXTENSIONS = {".txt", ".md", ".csv", ".html", ".htm", ".pdf", ".xlsx", ".xlsm", ".zip"}
CONTENT_TYPE_EXTENSIONS = {
    "application/pdf": ".pdf",
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
    "application/vnd.ms-excel.sheet.macroenabled.12": ".xlsm",
    "text/plain": ".txt",
    "text/csv": ".csv",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
}
METRIC_SYNONYMS: dict[str, list[str]] = {
    "report_revenue": [
        r"выручк[аи]",
        r"revenue",
        r"sales",
        r"net\s+sales",
        r"net\s+interest\s+income",
        r"чистые\s+процентные\s+доходы",
    ],
    "report_ebitda": [r"ebitda", r"oibda", r"операционн\w+\s+прибыл\w+\s+до\s+амортизац"],
    "report_net_profit": [r"чист\w+\s+прибыл\w+", r"net\s+profit", r"profit\s+for\s+the\s+period", r"net\s+income"],
    "report_operating_cash_flow": [
        r"операционн\w+\s+денежн\w+\s+поток",
        r"денежн\w+\s+поток\w+\s+от\s+операционн\w+\s+деятельност",
        r"cash\s+flows?\s+from\s+operating\s+activities",
        r"operating\s+cash\s+flow",
    ],
    "report_free_cash_flow": [r"свободн\w+\s+денежн\w+\s+поток", r"free\s+cash\s+flow", r"fcf"],
    "report_capex": [
        r"капитальн\w+\s+затрат",
        r"капитальн\w+\s+вложен",
        r"capital\s+expenditure",
        r"capex",
        r"additions\s+to\s+property",
    ],
    "report_total_debt": [
        r"общ\w+\s+долг",
        r"совокупн\w+\s+долг",
        r"заемн\w+\s+средств",
        r"total\s+debt",
        r"borrowings",
        r"loans\s+and\s+borrowings",
    ],
    "report_short_term_debt": [r"краткосрочн\w+\s+долг", r"short[-\s]?term\s+debt", r"current\s+borrowings"],
    "report_long_term_debt": [r"долгосрочн\w+\s+долг", r"long[-\s]?term\s+debt", r"non[-\s]?current\s+borrowings"],
    "report_cash_and_equivalents": [
        r"денежн\w+\s+средств\w+\s+и\s+их\s+эквивалент",
        r"cash\s+and\s+cash\s+equivalents",
        r"cash\s+and\s+equivalents",
    ],
    "report_interest_expense": [
        r"процентн\w+\s+расход",
        r"interest\s+expense",
        r"finance\s+costs?",
        r"финансов\w+\s+расход",
    ],
    "report_total_assets": [r"активы\s+итого", r"итого\s+актив", r"total\s+assets"],
    "report_total_equity": [r"капитал\s+итого", r"итого\s+капитал", r"total\s+equity", r"shareholders[’']?\s+equity"],
    "report_dividends": [r"дивиденд", r"dividends"],
    "report_provisions": [r"резерв\w+", r"provisions?"],
    "report_cet1_ratio": [r"норматив\s+достаточности\s+базового\s+капитала", r"cet1", r"tier\s+1\s+capital\s+ratio"],
}
TEXT_SIGNAL_PATTERNS: dict[str, list[str]] = {
    "sanctions": [r"санкци", r"sanction", r"restricted\s+market", r"market\s+access"],
    "currency_fx": [r"валютн\w+\s+риск", r"курсов\w+\s+разниц", r"foreign\s+exchange", r"currency\s+risk", r"fx\s+risk"],
    "liquidity_refinancing": [
        r"ликвидност\w+\s+риск",
        r"рефинансирован",
        r"refinancing",
        r"liquidity\s+risk",
        r"maturity\s+profile",
    ],
    "covenant": [r"ковенант", r"covenant"],
    "impairment": [r"обесценен", r"impairment"],
    "litigation": [r"судебн\w+\s+разбирательств", r"litigation", r"legal\s+claim"],
    "going_concern": [r"непрерывност\w+\s+деятельност", r"going\s+concern"],
    "dividend_pressure": [r"дивиденд\w+\s+не\s+выпла", r"приостанов\w+\s+дивиденд", r"dividend\s+suspend", r"no\s+dividend"],
    "tax_pressure": [r"налогов\w+\s+нагруз", r"tax\s+burden", r"income\s+tax\s+rate"],
    "capex_pressure": [r"рост\w+\s+капитальн\w+\s+затрат", r"capex\s+increase", r"investment\s+program"],
    "demand_pressure": [r"снижени\w+\s+спрос", r"demand\s+slowdown", r"lower\s+demand"],
    "auditor_emphasis": [r"модифицированн\w+\s+мнени", r"emphasis\s+of\s+matter", r"qualified\s+opinion"],
    "related_party": [r"связанн\w+\s+сторон", r"related\s+part"],
}

SCALE_RE = r"(?:трлн|триллион\w*|trillion|trn|млрд|миллиард\w*|billion|bn|млн|миллион\w*|million|mn|тыс|тысяч\w*|thousand|k)"
NUMBER_RE = r"\(?[-+−]?\d[\d\s\u00a0.,']*\)?"


@dataclass(frozen=True)
class ReportSource:
    ticker: str
    company_name: str
    e_disclosure_id: str | None = None
    issuer_url: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class ReportRecord:
    ticker: str
    company_name: str | None
    report_period_end: pd.Timestamp | None
    publish_date: pd.Timestamp | None
    report_type: str | None
    accounting_standard: str | None
    language: str | None
    source_url: str | None
    local_path: str | None
    source_name: str | None = None
    file_sha256: str | None = None


def _read_any_frame(path: str | Path) -> pd.DataFrame:
    return load_local_csv(path)


def _parse_date(value: Any) -> pd.Timestamp | pd.NaT:
    if value is None or (isinstance(value, float) and np.isnan(value)) or value == "":
        return pd.NaT
    if isinstance(value, pd.Timestamp):
        return value
    text = str(value).strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return pd.to_datetime(text, errors="coerce", dayfirst=False)
    return pd.to_datetime(text, errors="coerce", dayfirst=True)


def _supported_extension(ext: str | None) -> str | None:
    if not ext:
        return None
    ext = ext.lower().strip()
    if ext == ".xls":
        return None
    return ext if ext in SUPPORTED_REPORT_EXTENSIONS else None


def _extension_from_url(url: str | None) -> str | None:
    if not url:
        return None
    suffix = Path(unquote(urlparse(str(url)).path)).suffix.lower()
    return _supported_extension(suffix)


def _extension_from_content_disposition(header: str | None) -> str | None:
    if not header:
        return None
    match = re.search(r"filename\*=(?:UTF-8'')?([^;]+)", header, flags=re.IGNORECASE)
    if not match:
        match = re.search(r"filename=(\"[^\"]+\"|[^;]+)", header, flags=re.IGNORECASE)
    if not match:
        return None
    filename = unquote(match.group(1).strip().strip('"'))
    return _supported_extension(Path(filename).suffix)


def _extension_from_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    media_type = content_type.split(";", 1)[0].strip().lower()
    return CONTENT_TYPE_EXTENSIONS.get(media_type)


def _extension_from_magic_bytes(content: bytes) -> str | None:
    head = content[:8]
    if head.startswith(b"%PDF"):
        return ".pdf"
    if head.startswith(b"PK\x03\x04"):
        try:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                names = set(zf.namelist())
                if "[Content_Types].xml" in names and any(name.startswith("xl/") for name in names):
                    return ".xlsx"
        except Exception:
            pass
        return ".zip"
    if head.lstrip().lower().startswith((b"<!doctype", b"<html")):
        return ".html"
    return None


def _detect_report_extension(
    *,
    url: str | None = None,
    content_type: str | None = None,
    content_disposition: str | None = None,
    content: bytes | None = None,
    fallback: str = ".bin",
) -> str:
    """Detect a supported report extension from the strongest available signal."""
    for candidate in [
        _extension_from_content_disposition(content_disposition),
        _extension_from_content_type(content_type),
        _extension_from_magic_bytes(content or b""),
        _extension_from_url(url),
        _supported_extension(fallback),
    ]:
        if candidate:
            return candidate
    return fallback


def load_report_registry(path: str | Path | None = None, frame: pd.DataFrame | None = None) -> pd.DataFrame:
    """Load a report registry and normalize its canonical columns.

    Registry rows can point to local documents, direct URLs or already embedded
    text via the `report_text` column. The key point-in-time column is
    `publish_date`; reports without a publish date are not merged into the model
    panel until the date is provided or inferred by a collector.
    """
    if frame is not None:
        df = frame.copy()
    elif path is not None:
        df = _read_any_frame(path)
    else:
        raise ValueError("Either path or frame must be provided")

    aliases = {
        "secid": "ticker",
        "security": "ticker",
        "issuer": "company_name",
        "period_end": "report_period_end",
        "date": "report_period_end",
        "published_at": "publish_date",
        "publication_date": "publish_date",
        "url": "source_url",
        "path": "local_path",
    }
    df = df.rename(columns={k: v for k, v in aliases.items() if k in df.columns})
    if "ticker" not in df.columns:
        raise ValueError("Report registry must contain ticker/secid column")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    for col in ["company_name", "report_type", "accounting_standard", "language", "source_url", "local_path", "source_name"]:
        if col not in df.columns:
            df[col] = None
    for col in ["report_period_end", "publish_date", "downloaded_at"]:
        if col in df.columns:
            df[col] = df[col].map(_parse_date)
    if "report_period_end" not in df.columns:
        df["report_period_end"] = pd.NaT
    if "publish_date" not in df.columns:
        df["publish_date"] = pd.NaT
    return df


def load_report_sources(path: str | Path) -> pd.DataFrame:
    df = _read_any_frame(path)
    if "ticker" not in df.columns:
        raise ValueError("Report source registry must contain ticker")
    df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    if "e_disclosure_id" not in df.columns:
        df["e_disclosure_id"] = None
    if "issuer_url" not in df.columns:
        df["issuer_url"] = None
    return df


def _scale_multiplier(scale_text: str | None, default: float = 1.0) -> float:
    if not scale_text:
        return default
    s = scale_text.lower()
    if re.search(r"трлн|триллион|trillion|trn", s):
        return 1e12
    if re.search(r"млрд|миллиард|billion|bn", s):
        return 1e9
    if re.search(r"млн|миллион|million|mn", s):
        return 1e6
    if re.search(r"тыс|тысяч|thousand|\bk\b", s):
        return 1e3
    return default


def parse_financial_number(raw: str | None, *, scale_text: str | None = None, default_multiplier: float = 1.0) -> float:
    """Parse Russian/English formatted financial numbers.

    Handles spaces as thousands separators, comma decimals, apostrophe thousands
    separators, unicode minus and parentheses for negative values.
    """
    if raw is None:
        return np.nan
    text = str(raw).strip().replace("\u00a0", " ").replace("−", "-")
    neg = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    text = re.sub(r"[^0-9,.'\-\s]", "", text)
    text = text.replace("'", "").replace(" ", "")
    if not text or text in {"-", ",", "."}:
        return np.nan

    if "," in text and "." in text:
        comma = text.rfind(",")
        dot = text.rfind(".")
        if comma > dot:
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        tail = text.rsplit(",", 1)[-1]
        text = text.replace(",", ".") if 0 < len(tail) <= 2 else text.replace(",", "")
    elif text.count(".") > 1:
        text = text.replace(".", "")

    try:
        value = float(text)
    except ValueError:
        return np.nan
    if neg:
        value = -abs(value)
    return value * _scale_multiplier(scale_text, default_multiplier)


def _normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\t", " ")
    text = re.sub(r"[ ]+", " ", text)
    return text


def _extract_pdf_text(path: Path) -> str:
    try:
        from pypdf import PdfReader  # type: ignore
    except Exception as exc:  # pragma: no cover - ветка опциональной зависимости
        raise RuntimeError("pypdf is required to extract PDF text; install pypdf or provide pre-extracted .txt/.md") from exc
    reader = PdfReader(str(path))
    chunks: list[str] = []
    for page in reader.pages:
        try:
            chunks.append(page.extract_text() or "")
        except Exception as exc:  # pragma: no cover - ветка битого pdf
            LOGGER.warning("PDF page extraction failed for %s: %s", path, exc)
    return "\n".join(chunks)


def _extract_xlsx_text(path: Path) -> str:
    try:
        import openpyxl  # type: ignore
    except Exception as exc:  # pragma: no cover - ветка опциональной зависимости
        raise RuntimeError("openpyxl is required to extract XLSX report tables") from exc
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    chunks: list[str] = []
    for ws in wb.worksheets:
        chunks.append(f"SHEET {ws.title}")
        for row in ws.iter_rows(values_only=True):
            values = [str(v) for v in row if v is not None]
            if values:
                chunks.append(" | ".join(values))
    return "\n".join(chunks)


def extract_text_from_path(path: str | Path) -> str:
    """Extract text from txt/md/html/pdf/xlsx or from files inside a zip archive."""
    path = Path(path)
    suffix = _supported_extension(path.suffix)
    if suffix is None and path.exists():
        suffix = _extension_from_magic_bytes(path.read_bytes())
    if suffix in {".txt", ".md", ".csv", ".html", ".htm"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if suffix == ".pdf":
        return _extract_pdf_text(path)
    if suffix in {".xlsx", ".xlsm"}:
        return _extract_xlsx_text(path)
    if suffix == ".zip":
        chunks: list[str] = []
        with zipfile.ZipFile(path) as zf:
            for name in sorted(zf.namelist()):
                inner_suffix = Path(name).suffix.lower()
                if inner_suffix not in {".txt", ".md", ".html", ".htm", ".pdf", ".xlsx", ".xlsm"}:
                    continue
                tmp_dir = path.parent / f".{path.stem}_unzipped"
                tmp_dir.mkdir(exist_ok=True)
                extracted = Path(zf.extract(name, tmp_dir))
                try:
                    chunks.append(extract_text_from_path(extracted))
                except Exception as exc:
                    LOGGER.warning("Could not extract %s from %s: %s", name, path, exc)
        return "\n".join(chunks)
    raise ValueError(f"Unsupported report file format: {path}")


def _candidate_numbers_after_keyword(text: str, keyword_regex: str, *, default_multiplier: float) -> list[tuple[float, str]]:
    pattern = re.compile(keyword_regex, flags=re.IGNORECASE | re.MULTILINE)
    candidates: list[tuple[float, str]] = []
    for match in pattern.finditer(text):
        start = match.end()
        window = text[start : start + 260]
        for num_match in re.finditer(rf"(?P<num>{NUMBER_RE})\s*(?P<scale>{SCALE_RE})?", window, flags=re.IGNORECASE):
            raw = num_match.group("num")
            scale = num_match.group("scale")
            value = parse_financial_number(raw, scale_text=scale, default_multiplier=default_multiplier)
            if np.isnan(value):
                continue
            if scale is None and 1900 <= abs(value) <= 2100 and float(value).is_integer():
                continue
            evidence = text[max(0, match.start() - 80) : min(len(text), match.start() + 260)]
            candidates.append((value, evidence.strip()))
    return candidates


def _candidate_numbers_before_keyword(text: str, keyword_regex: str, *, default_multiplier: float) -> list[tuple[float, str]]:
    pattern = re.compile(keyword_regex, flags=re.IGNORECASE | re.MULTILINE)
    candidates: list[tuple[float, str]] = []
    for match in pattern.finditer(text):
        window = text[max(0, match.start() - 180) : match.start()]
        numbers = list(re.finditer(rf"(?P<num>{NUMBER_RE})\s*(?P<scale>{SCALE_RE})?", window, flags=re.IGNORECASE))
        for num_match in numbers[-3:]:
            raw = num_match.group("num")
            scale = num_match.group("scale")
            value = parse_financial_number(raw, scale_text=scale, default_multiplier=default_multiplier)
            if np.isnan(value):
                continue
            if scale is None and 1900 <= abs(value) <= 2100 and float(value).is_integer():
                continue
            evidence = text[max(0, match.start() - 180) : min(len(text), match.end() + 80)]
            candidates.append((value, evidence.strip()))
    return candidates


def _select_best_metric_candidate(candidates: list[tuple[float, str]], metric: str) -> tuple[float, str | None]:
    if not candidates:
        return np.nan, None
    for value, evidence in candidates:
        if abs(value) > NUMERIC_EPS:
            return float(value), evidence
    return float(candidates[0][0]), candidates[0][1]


def extract_numeric_metrics_from_text(text: str, *, default_multiplier: float = 1.0) -> tuple[dict[str, float], dict[str, str]]:
    text = _normalize_text(text)
    values: dict[str, float] = {}
    evidence: dict[str, str] = {}
    for metric, synonyms in METRIC_SYNONYMS.items():
        all_candidates: list[tuple[float, str]] = []
        for synonym in synonyms:
            all_candidates.extend(_candidate_numbers_after_keyword(text, synonym, default_multiplier=default_multiplier))
            all_candidates.extend(_candidate_numbers_before_keyword(text, synonym, default_multiplier=default_multiplier))
        value, ev = _select_best_metric_candidate(all_candidates, metric)
        values[metric] = value
        if ev:
            evidence[f"{metric}_evidence"] = ev[:700]
    return values, evidence


def extract_text_signals(text: str) -> dict[str, float]:
    clean = _normalize_text(text).lower()
    out: dict[str, float] = {
        "report_text_length": float(len(clean)),
        "report_word_count": float(len(re.findall(r"\w+", clean))),
    }
    total_count = 0
    for name, patterns in TEXT_SIGNAL_PATTERNS.items():
        count = 0
        for pattern in patterns:
            count += len(re.findall(pattern, clean, flags=re.IGNORECASE))
        out[f"report_{name}_count"] = float(count)
        out[f"report_{name}_flag"] = float(count > 0)
        total_count += count
    out["report_text_risk_terms_total"] = float(total_count)
    out["report_text_risk_density"] = float(total_count / max(out["report_word_count"], 1.0))
    return out


def _safe_divide(a: Any, b: Any) -> float:
    try:
        a = float(a)
        b = float(b)
    except Exception:
        return np.nan
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < NUMERIC_EPS:
        return np.nan
    return a / b


def compute_report_ratios(row: dict[str, Any]) -> dict[str, float]:
    revenue = row.get("report_revenue")
    ebitda = row.get("report_ebitda")
    net_profit = row.get("report_net_profit")
    ocf = row.get("report_operating_cash_flow")
    fcf = row.get("report_free_cash_flow")
    capex = row.get("report_capex")
    total_debt = row.get("report_total_debt")
    cash = row.get("report_cash_and_equivalents")
    interest = row.get("report_interest_expense")
    total_assets = row.get("report_total_assets")
    equity = row.get("report_total_equity")
    short_debt = row.get("report_short_term_debt")
    fcf_proxy = fcf
    if not np.isfinite(fcf_proxy) if isinstance(fcf_proxy, float) else pd.isna(fcf_proxy):
        if pd.notna(ocf) and pd.notna(capex):
            fcf_proxy = float(ocf) - abs(float(capex))

    net_debt = np.nan
    if pd.notna(total_debt) and pd.notna(cash):
        net_debt = float(total_debt) - float(cash)

    ratios = {
        "report_net_debt": net_debt,
        "report_net_debt_to_ebitda": _safe_divide(net_debt, ebitda),
        "report_interest_coverage": _safe_divide(ebitda, abs(float(interest)) if pd.notna(interest) else np.nan),
        "report_ebitda_margin": _safe_divide(ebitda, revenue),
        "report_net_margin": _safe_divide(net_profit, revenue),
        "report_operating_cf_margin": _safe_divide(ocf, revenue),
        "report_free_cf_margin": _safe_divide(fcf_proxy, revenue),
        "report_capex_intensity": _safe_divide(abs(float(capex)) if pd.notna(capex) else np.nan, revenue),
        "report_equity_ratio": _safe_divide(equity, total_assets),
        "report_debt_to_assets": _safe_divide(total_debt, total_assets),
        "report_short_debt_share": _safe_divide(short_debt, total_debt),
    }
    return {k: (float(v) if pd.notna(v) and np.isfinite(v) else np.nan) for k, v in ratios.items()}


def _quality_score(row: dict[str, Any]) -> float:
    metrics_found = sum(pd.notna(row.get(k)) for k in METRIC_SYNONYMS)
    ratios_found = sum(pd.notna(row.get(k)) for k in [
        "report_net_debt_to_ebitda",
        "report_interest_coverage",
        "report_ebitda_margin",
        "report_free_cf_margin",
        "report_equity_ratio",
    ])
    text_words = float(row.get("report_word_count") or 0.0)
    text_bonus = min(text_words / 5000.0, 1.0)
    return float(min(1.0, metrics_found / 10.0 * 0.55 + ratios_found / 5.0 * 0.30 + text_bonus * 0.15))


def extract_financial_report_features(
    text: str,
    *,
    default_multiplier: float = 1.0,
    include_evidence: bool = True,
) -> dict[str, Any]:
    metrics, evidence = extract_numeric_metrics_from_text(text, default_multiplier=default_multiplier)
    signals = extract_text_signals(text)
    row: dict[str, Any] = {**metrics, **signals}
    row.update(compute_report_ratios(row))
    leverage = row.get("report_net_debt_to_ebitda")
    leverage_component = 0.0 if pd.isna(leverage) else float(np.clip(leverage, -2, 8) / 8.0)
    fcf_margin = row.get("report_free_cf_margin")
    fcf_component = 0.0 if pd.isna(fcf_margin) else float(np.clip(-fcf_margin, -1, 1))
    coverage = row.get("report_interest_coverage")
    coverage_component = 0.0 if pd.isna(coverage) else float(np.clip(1.0 / (1.0 + max(float(coverage), 0.0)), 0, 1))
    text_component = float(np.clip(row.get("report_text_risk_density", 0.0) * 100.0, 0, 1))
    row["report_financial_pressure"] = float(0.35 * leverage_component + 0.25 * fcf_component + 0.20 * coverage_component + 0.20 * text_component)
    row["report_extraction_quality"] = _quality_score(row)

    for metric in METRIC_SYNONYMS:
        row[f"{metric}_missing"] = float(pd.isna(row.get(metric)))
    if include_evidence:
        row.update(evidence)
    return row


def _resolve_local_report_path(local_path: Any, *, registry_path: str | Path | None = None, reports_dir: str | Path | None = None) -> Path | None:
    if pd.isna(local_path) or local_path is None or str(local_path).strip() == "":
        return None
    path = Path(str(local_path))
    candidates = []
    if path.is_absolute():
        candidates.append(path)
    if reports_dir is not None:
        candidates.append(Path(reports_dir) / path)
    if registry_path is not None:
        candidates.append(Path(registry_path).parent / path)
    candidates.append(path)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build_financial_report_features(
    registry: str | Path | pd.DataFrame,
    *,
    reports_dir: str | Path | None = None,
    registry_path: str | Path | None = None,
    default_multiplier: float = 1.0,
    include_evidence: bool = True,
) -> pd.DataFrame:
    """Build deterministic features from financial reports.

    The function accepts either a path or a DataFrame registry. Each row can use
    `report_text` directly or refer to a local document. Rows without text are
    retained with missing indicators so coverage gaps are explicit.
    """
    if isinstance(registry, (str, Path)):
        registry_path = registry_path or registry
        reg = load_report_registry(registry)
    else:
        reg = load_report_registry(frame=registry)
    rows: list[dict[str, Any]] = []

    for idx, rec in reg.iterrows():
        base: dict[str, Any] = {
            "ticker": str(rec.get("ticker", "")).upper().strip(),
            "company_name": rec.get("company_name"),
            "report_period_end": _parse_date(rec.get("report_period_end")),
            "publish_date": _parse_date(rec.get("publish_date")),
            "report_type": rec.get("report_type"),
            "accounting_standard": rec.get("accounting_standard"),
            "language": rec.get("language"),
            "source_url": rec.get("source_url"),
            "source_name": rec.get("source_name"),
            "local_path": rec.get("local_path"),
            "discovery_status": rec.get("discovery_status"),
            "discovery_error": rec.get("discovery_error"),
            "download_status": rec.get("download_status"),
            "content_type": rec.get("content_type"),
            "detected_extension": rec.get("detected_extension"),
            "parse_status": "not_attempted",
            "parse_error": None,
        }
        text = rec.get("report_text") if "report_text" in reg.columns else None
        text_source = "embedded_text" if text is not None and not pd.isna(text) and str(text).strip() else None
        path = _resolve_local_report_path(rec.get("local_path"), registry_path=registry_path, reports_dir=reports_dir)
        if (text is None or pd.isna(text) or str(text).strip() == "") and path is not None and path.exists():
            try:
                text = extract_text_from_path(path)
                base["file_sha256"] = _file_sha256(path)
                text_source = "file"
                if not str(text).strip():
                    base["parse_status"] = "empty_text"
            except Exception as exc:
                LOGGER.warning("Report text extraction failed for row=%s path=%s: %s", idx, path, exc)
                text = ""
                base["parse_status"] = "extract_failed"
                base["parse_error"] = str(exc)[:500]
        elif path is not None and path.exists():
            base["file_sha256"] = _file_sha256(path)
        else:
            text = "" if text is None or pd.isna(text) else str(text)
            if not str(text).strip():
                base["parse_status"] = "missing_document"

        if text and str(text).strip():
            try:
                features = extract_financial_report_features(str(text), default_multiplier=default_multiplier, include_evidence=include_evidence)
                base.update(features)
                base["report_document_available"] = 1.0
                base["parse_status"] = f"parsed_{text_source or 'text'}"
            except Exception as exc:
                LOGGER.warning("Report feature extraction failed for row=%s: %s", idx, exc)
                base["parse_status"] = "parse_failed"
                base["parse_error"] = str(exc)[:500]
                text = ""
        if not text or not str(text).strip():
            if base["parse_status"] == "not_attempted":
                base["parse_status"] = "missing_document"
            missing_features = {metric: np.nan for metric in METRIC_SYNONYMS}
            missing_features.update({f"{metric}_missing": 1.0 for metric in METRIC_SYNONYMS})
            missing_features.update({
                "report_text_length": 0.0,
                "report_word_count": 0.0,
                "report_text_risk_terms_total": 0.0,
                "report_text_risk_density": 0.0,
                "report_financial_pressure": np.nan,
                "report_extraction_quality": 0.0,
                "report_document_available": 0.0,
            })
            for name in TEXT_SIGNAL_PATTERNS:
                missing_features[f"report_{name}_count"] = 0.0
                missing_features[f"report_{name}_flag"] = 0.0
            missing_features.update(compute_report_ratios(missing_features))
            base.update(missing_features)
        else:
            base["parse_error"] = None
        rows.append(base)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for col in ["report_period_end", "publish_date"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")

    numeric_cols = [c for c in out.columns if c.startswith("report_") and not c.endswith("_evidence") and c not in {"report_type", "report_period_end"}]
    for col in numeric_cols:
        converted = pd.to_numeric(out[col], errors="coerce")
        if converted.notna().sum() > 0 or out[col].isna().all():
            out[col] = converted
    out = out.sort_values(["ticker", "report_period_end", "publish_date"], na_position="last").reset_index(drop=True)
    yoy_metrics = [
        "report_revenue",
        "report_ebitda",
        "report_net_profit",
        "report_operating_cash_flow",
        "report_free_cash_flow",
        "report_total_debt",
        "report_net_debt",
        "report_cash_and_equivalents",
    ]
    for col in yoy_metrics:
        if col in out.columns:
            prev = out.groupby("ticker")[col].shift(1)
            out[f"{col}_yoy_change"] = (out[col] - prev) / prev.abs().replace(0, np.nan)
    return out


def report_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "ticker",
        "company_name",
        "report_period_end",
        "publish_date",
        "report_type",
        "accounting_standard",
        "language",
        "source_url",
        "source_name",
        "local_path",
        "file_sha256",
        "discovery_status",
        "discovery_error",
        "download_status",
        "content_type",
        "detected_extension",
        "parse_status",
        "parse_error",
    }
    return [c for c in df.columns if c not in excluded and not c.endswith("_evidence")]


def merge_financial_report_features_pit(
    panel: pd.DataFrame,
    report_features: pd.DataFrame,
    *,
    date_col: str = "decision_date",
    ticker_col: str = "ticker",
    max_lag_days: int | None = 550,
) -> pd.DataFrame:
    """As-of merge report features into a model panel by publish_date.

    A report can influence a decision only if `publish_date <= decision_date`.
    This is the central leakage-control rule for the financial-report layer.
    """
    data = panel.copy()
    if data.empty or report_features is None or report_features.empty:
        data["report_available"] = 0.0
        return data
    if date_col not in data.columns or ticker_col not in data.columns:
        raise ValueError(f"Panel must contain {date_col} and {ticker_col}")
    reports = report_features.copy()
    if "publish_date" not in reports.columns or "ticker" not in reports.columns:
        raise ValueError("report_features must contain ticker and publish_date")

    data[date_col] = pd.to_datetime(data[date_col], errors="coerce")
    data[ticker_col] = data[ticker_col].astype(str).str.upper().str.strip()
    reports["publish_date"] = pd.to_datetime(reports["publish_date"], errors="coerce")
    reports["ticker"] = reports["ticker"].astype(str).str.upper().str.strip()
    reports = reports.dropna(subset=["publish_date", "ticker"]).sort_values(["ticker", "publish_date"])

    out_frames: list[pd.DataFrame] = []
    right_cols = ["ticker", "publish_date", "report_period_end"] + report_feature_columns(reports)
    right_cols = [c for c in right_cols if c in reports.columns]
    for ticker, left in data.sort_values([ticker_col, date_col]).groupby(ticker_col, sort=False):
        right = reports[reports["ticker"] == ticker][right_cols].sort_values("publish_date")
        left_sorted = left.sort_values(date_col)
        if right.empty:
            merged = left_sorted.copy()
            merged["publish_date_report"] = pd.NaT
        else:
            merged = pd.merge_asof(
                left_sorted,
                right,
                left_on=date_col,
                right_on="publish_date",
                direction="backward",
                suffixes=("", "_report"),
            )
            if "ticker_report" in merged.columns:
                merged = merged.drop(columns=["ticker_report"])
            merged = merged.rename(columns={"publish_date": "publish_date_report"})
        out_frames.append(merged)
    merged = pd.concat(out_frames, ignore_index=True).sort_values([date_col, ticker_col]).reset_index(drop=True)
    merged["report_available"] = merged.get("report_document_available", pd.Series(0, index=merged.index)).fillna(0).astype(float)
    if "publish_date_report" in merged.columns:
        lag = (merged[date_col] - pd.to_datetime(merged["publish_date_report"], errors="coerce")).dt.days
        merged["report_lag_days"] = lag
        merged["report_stale_flag"] = 0.0
        if max_lag_days is not None:
            stale = lag.isna() | (lag > max_lag_days)
            merged["report_stale_flag"] = stale.astype(float)
            feature_cols = [c for c in report_feature_columns(report_features) if c in merged.columns]
            if feature_cols:
                merged.loc[stale, feature_cols] = np.nan
                merged.loc[stale, "report_available"] = 0.0
    else:
        merged["report_lag_days"] = np.nan
        merged["report_stale_flag"] = 1.0
    return merged


def _request_text(url: str, params: dict[str, Any] | None = None, retries: int = 3, sleep: float = 1.0) -> str:
    headers = {
        "User-Agent": "risk-pipeline-research/0.3 (+academic reproducibility)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(url, params=params, headers=headers, timeout=30)
            response.raise_for_status()
            response.encoding = response.encoding or "utf-8"
            return response.text
        except requests.HTTPError as exc:  # pragma: no cover - зависит от сети
            status = exc.response.status_code if exc.response is not None else None
            if status in {401, 403, 404}:
                raise DownloadError(f"Could not download {url}: HTTP {status}") from exc
            last_error = exc
            LOGGER.warning("request failed: attempt=%s url=%s error=%s", attempt, url, exc)
            time.sleep(sleep * attempt)
        except Exception as exc:  # pragma: no cover - зависит от сети
            last_error = exc
            LOGGER.warning("request failed: attempt=%s url=%s error=%s", attempt, url, exc)
            time.sleep(sleep * attempt)
    raise DownloadError(f"Could not download {url}: {last_error}")


def _parse_russian_date_from_text(text: str) -> pd.Timestamp | pd.NaT:
    match = re.search(r"(\d{2}\.\d{2}\.\d{4})", text or "")
    if not match:
        return pd.NaT
    return pd.to_datetime(match.group(1), errors="coerce", dayfirst=True)


def _infer_period_end(text: str, report_year: int | None = None) -> pd.Timestamp | pd.NaT:
    lower = (text or "").lower()
    year_match = re.search(r"(20\d{2})", lower)
    year = report_year or (int(year_match.group(1)) if year_match else None)
    if not year:
        return pd.NaT
    if re.search(r"(3\s*месяц|1\s*кв|q1|first\s+quarter)", lower):
        return pd.Timestamp(year=year, month=3, day=31)
    if re.search(r"(6\s*месяц|2\s*кв|q2|half[-\s]?year|six\s+months)", lower):
        return pd.Timestamp(year=year, month=6, day=30)
    if re.search(r"(9\s*месяц|3\s*кв|q3|nine\s+months)", lower):
        return pd.Timestamp(year=year, month=9, day=30)
    return pd.Timestamp(year=year, month=12, day=31)


def discover_e_disclosure_reports(
    ticker: str,
    company_name: str,
    e_disclosure_id: str | int,
    *,
    document_types: Iterable[int] = (4, 2),
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Discover report links from Interfax e-disclosure pages.

    `type=4` corresponds to consolidated IFRS-style financial statements on the
    portal, and `type=2` to annual reports. The parser is conservative and stores
    the raw row text for audit because the portal markup may change.
    """
    start_ts = pd.to_datetime(start) if start else None
    end_ts = pd.to_datetime(end) if end else None
    rows: list[dict[str, Any]] = []
    for doc_type in document_types:
        html = _request_text(E_DISCLOSURE_FILES_URL, params={"id": str(e_disclosure_id), "type": str(doc_type)})  # pragma: no cover - зависит от сети
        for match in re.finditer(r"(?is)<tr[^>]*>(.*?)</tr>", html):
            row_html = match.group(1)
            if "FileLoad.ashx" not in row_html and "fileload.ashx" not in row_html.lower():
                continue
            text = re.sub(r"<[^>]+>", " ", row_html)
            text = re.sub(r"\s+", " ", text).strip()
            href_match = re.search(r"href=[\"']([^\"']*FileLoad\.ashx[^\"']*)[\"']", row_html, flags=re.IGNORECASE)
            href = urljoin(E_DISCLOSURE_BASE, href_match.group(1).replace("&amp;", "&")) if href_match else None
            pub_dates = re.findall(r"\d{2}\.\d{2}\.\d{4}", text)
            publish_date = pd.to_datetime(pub_dates[-1], errors="coerce", dayfirst=True) if pub_dates else pd.NaT
            year_match = re.search(r"\b(20\d{2})\b", text)
            year = int(year_match.group(1)) if year_match else None
            period_end = _infer_period_end(text, year)
            if start_ts is not None and pd.notna(publish_date) and publish_date < start_ts:
                continue
            if end_ts is not None and pd.notna(publish_date) and publish_date > end_ts:
                continue
            rows.append(
                {
                    "ticker": ticker.upper(),
                    "company_name": company_name,
                    "report_period_end": period_end,
                    "publish_date": publish_date,
                    "report_type": "ifrs_consolidated" if doc_type == 4 else "annual_report",
                    "accounting_standard": "IFRS" if doc_type == 4 else None,
                    "language": "ru",
                    "source_name": "e-disclosure",
                    "source_url": href,
                    "local_path": None,
                    "e_disclosure_id": str(e_disclosure_id),
                    "raw_row_text": text,
                }
            )
    return pd.DataFrame(rows)


def discover_reports_for_sources(
    sources: str | Path | pd.DataFrame,
    *,
    start: str | None = None,
    end: str | None = None,
    document_types: Iterable[int] = (4, 2),
) -> pd.DataFrame:
    if isinstance(sources, (str, Path)):
        src = load_report_sources(sources)
    else:
        src = sources.copy()
        src["ticker"] = src["ticker"].astype(str).str.upper().str.strip()
    frames: list[pd.DataFrame] = []
    for _, row in src.iterrows():
        ed_id = row.get("e_disclosure_id")
        if pd.notna(ed_id) and str(ed_id).strip():
            try:
                discovered = discover_e_disclosure_reports(
                    row["ticker"],
                    row.get("company_name") or row["ticker"],
                    str(ed_id),
                    document_types=document_types,
                    start=start,
                    end=end,
                )
                if not discovered.empty:
                    discovered["discovery_status"] = "ok"
                    discovered["discovery_error"] = None
                    frames.append(discovered)
            except Exception as exc:  # pragma: no cover - зависит от сети
                LOGGER.warning("E-disclosure discovery failed for %s id=%s: %s", row["ticker"], ed_id, exc)
                frames.append(
                    pd.DataFrame(
                        [
                            {
                                "ticker": row["ticker"],
                                "company_name": row.get("company_name"),
                                "report_period_end": pd.NaT,
                                "publish_date": pd.NaT,
                                "report_type": "e_disclosure_discovery",
                                "accounting_standard": None,
                                "language": None,
                                "source_name": "e-disclosure",
                                "source_url": f"{E_DISCLOSURE_FILES_URL}?id={ed_id}",
                                "local_path": None,
                                "e_disclosure_id": ed_id,
                                "raw_row_text": None,
                                "discovery_status": "failed",
                                "discovery_error": str(exc)[:500],
                            }
                        ]
                    )
                )
        issuer_url = row.get("issuer_url")
        if pd.notna(issuer_url) and str(issuer_url).strip():
            frames.append(
                pd.DataFrame(
                    [
                        {
                            "ticker": row["ticker"],
                            "company_name": row.get("company_name"),
                            "report_period_end": pd.NaT,
                            "publish_date": pd.NaT,
                            "report_type": "issuer_ir_page",
                            "accounting_standard": None,
                            "language": None,
                            "source_name": "issuer_ir",
                            "source_url": issuer_url,
                            "local_path": None,
                            "e_disclosure_id": ed_id,
                            "raw_row_text": "issuer investor-relations source page",
                            "discovery_status": "manual_fallback",
                            "discovery_error": None,
                        }
                    ]
                )
            )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True).drop_duplicates(subset=["ticker", "source_url", "report_type"], keep="first")


def download_report_registry(
    registry: str | Path | pd.DataFrame,
    *,
    out_dir: str | Path,
    registry_output: str | Path | None = None,
    skip_existing: bool = True,
    sleep: float = 0.7,
) -> pd.DataFrame:
    """Download direct report URLs from a registry.

    The function is deliberately transparent: it records HTTP status, local file,
    SHA-256 and failures in the returned registry. It does not silently drop rows.
    """
    reg = load_report_registry(registry) if isinstance(registry, (str, Path)) else load_report_registry(frame=registry)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "risk-pipeline-research/0.3 (+academic reproducibility)"}
    statuses: list[str] = []
    local_paths: list[str | None] = []
    hashes: list[str | None] = []
    content_types: list[str | None] = []
    detected_extensions: list[str | None] = []

    for idx, row in reg.iterrows():
        url = row.get("source_url")
        ticker = str(row.get("ticker", "UNKNOWN")).upper()
        if pd.isna(url) or not str(url).strip() or "files.aspx" in str(url).lower():
            statuses.append("no_direct_file_url")
            local_paths.append(row.get("local_path"))
            hashes.append(row.get("file_sha256") if "file_sha256" in reg.columns else None)
            content_types.append(row.get("content_type") if "content_type" in reg.columns else None)
            detected_extensions.append(row.get("detected_extension") if "detected_extension" in reg.columns else _supported_extension(Path(str(row.get("local_path") or "")).suffix))
            continue

        existing_path = _resolve_local_report_path(row.get("local_path"), reports_dir=out_dir)
        if skip_existing and existing_path is not None and existing_path.exists():
            statuses.append("existing")
            local_paths.append(row.get("local_path"))
            hashes.append(_file_sha256(existing_path))
            content_types.append(row.get("content_type") if "content_type" in reg.columns else None)
            detected_extensions.append(row.get("detected_extension") if "detected_extension" in reg.columns else _supported_extension(existing_path.suffix))
            continue

        date_part = "unknown"
        if pd.notna(row.get("report_period_end")):
            date_part = pd.to_datetime(row["report_period_end"]).strftime("%Y%m%d")
        try:  # pragma: no cover - зависит от сети
            response = requests.get(str(url), headers=headers, timeout=60)
            response.raise_for_status()
            content_type = response.headers.get("Content-Type")
            detected_ext = _detect_report_extension(
                url=str(url),
                content_type=content_type,
                content_disposition=response.headers.get("Content-Disposition"),
                content=response.content,
            )
            file_name = re.sub(r"[^A-Z0-9_\-]+", "_", f"{ticker}_{date_part}_{row.get('report_type') or 'report'}") + detected_ext
            rel_path = Path(ticker) / file_name
            path = out_dir / rel_path
            path.parent.mkdir(parents=True, exist_ok=True)
            if skip_existing and path.exists():
                statuses.append("existing")
                local_paths.append(str(rel_path))
                hashes.append(_file_sha256(path))
                content_types.append(content_type)
                detected_extensions.append(detected_ext)
                continue
            path.write_bytes(response.content)
            statuses.append(f"downloaded:{response.status_code}")
            local_paths.append(str(rel_path))
            hashes.append(_file_sha256(path))
            content_types.append(content_type)
            detected_extensions.append(detected_ext)
            time.sleep(sleep)
        except Exception as exc:  # pragma: no cover - зависит от сети
            LOGGER.warning("Download failed for %s: %s", url, exc)
            statuses.append(f"failed:{exc}")
            local_paths.append(row.get("local_path"))
            hashes.append(None)
            content_types.append(None)
            detected_extensions.append(_extension_from_url(str(url)))
    reg = reg.copy()
    reg["download_status"] = statuses
    reg["local_path"] = local_paths
    reg["file_sha256"] = hashes
    reg["content_type"] = content_types
    reg["detected_extension"] = detected_extensions
    if registry_output:
        save_frame(reg, registry_output)
    return reg


def coverage_report(panel: pd.DataFrame, merged_panel: pd.DataFrame, *, date_col: str = "decision_date", ticker_col: str = "ticker") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker, group in merged_panel.groupby(ticker_col):
        avail = pd.to_numeric(group.get("report_available", pd.Series(0, index=group.index)), errors="coerce").fillna(0)
        rows.append(
            {
                "ticker": ticker,
                "observations": int(len(group)),
                "report_covered_observations": int((avail > 0).sum()),
                "coverage_rate": float((avail > 0).mean()) if len(group) else 0.0,
                "first_decision_date": pd.to_datetime(group[date_col]).min(),
                "last_decision_date": pd.to_datetime(group[date_col]).max(),
                "median_report_lag_days": float(pd.to_numeric(group.get("report_lag_days"), errors="coerce").median()) if "report_lag_days" in group else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values(["coverage_rate", "ticker"], ascending=[False, True])


def write_report_layer_summary(report_features: pd.DataFrame, merged_panel: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "report_rows": int(len(report_features)),
        "issuers_with_reports": int(report_features["ticker"].nunique()) if not report_features.empty and "ticker" in report_features else 0,
        "model_rows": int(len(merged_panel)),
        "covered_model_rows": int(pd.to_numeric(merged_panel.get("report_available", pd.Series(0, index=merged_panel.index)), errors="coerce").fillna(0).gt(0).sum()),
        "features": report_feature_columns(report_features) if not report_features.empty else [],
    }
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
