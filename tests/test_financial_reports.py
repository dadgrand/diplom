import pandas as pd

from risk_pipeline.financial_reports import (
    build_financial_report_features,
    discover_reports_for_sources,
    download_report_registry,
    extract_financial_report_features,
    merge_financial_report_features_pit,
    parse_financial_number,
    _detect_report_extension,
)


def test_parse_financial_number_russian_and_english_formats():
    assert parse_financial_number("1 234,5", scale_text="млн") == 1_234_500_000
    assert parse_financial_number("(2.5)", scale_text="billion") == -2_500_000_000
    assert parse_financial_number("3,250.75") == 3250.75


def test_detect_report_extension_prefers_headers_over_ashx_url():
    ext = _detect_report_extension(
        url="https://www.e-disclosure.ru/portal/FileLoad.ashx?Fileid=123",
        content_type="application/octet-stream",
        content_disposition='attachment; filename="issuer_report.pdf"',
        content=b"%PDF-1.7",
    )
    assert ext == ".pdf"


def test_download_report_registry_records_file_type_audit(tmp_path, monkeypatch):
    class FakeResponse:
        status_code = 200
        content = b"%PDF-1.7 fake report"
        headers = {
            "Content-Type": "application/octet-stream",
            "Content-Disposition": 'attachment; filename="ifrs_report.pdf"',
        }

        def raise_for_status(self):
            return None

    def fake_get(url, headers=None, timeout=None):
        return FakeResponse()

    monkeypatch.setattr("risk_pipeline.financial_reports.requests.get", fake_get)
    registry = pd.DataFrame(
        [
            {
                "ticker": "SBER",
                "report_period_end": "2024-12-31",
                "publish_date": "2025-03-01",
                "report_type": "ifrs",
                "source_url": "https://www.e-disclosure.ru/portal/FileLoad.ashx?Fileid=123",
            }
        ]
    )
    out = download_report_registry(registry, out_dir=tmp_path, sleep=0)
    assert out.loc[0, "detected_extension"] == ".pdf"
    assert out.loc[0, "content_type"] == "application/octet-stream"
    assert out.loc[0, "download_status"] == "downloaded:200"
    assert out.loc[0, "local_path"].endswith(".pdf")
    assert (tmp_path / out.loc[0, "local_path"]).exists()


def test_discover_reports_keeps_failure_rows_and_manual_fallback(monkeypatch):
    def fail_discovery(*args, **kwargs):
        raise RuntimeError("blocked")

    monkeypatch.setattr("risk_pipeline.financial_reports.discover_e_disclosure_reports", fail_discovery)
    sources = pd.DataFrame(
        [
            {
                "ticker": "SBER",
                "company_name": "Sber",
                "e_disclosure_id": "3043",
                "issuer_url": "https://example.test/ir",
            }
        ]
    )
    discovered = discover_reports_for_sources(sources)
    assert set(discovered["discovery_status"]) == {"failed", "manual_fallback"}
    assert "blocked" in discovered.loc[discovered["discovery_status"] == "failed", "discovery_error"].iloc[0]


def test_extract_financial_report_features_from_russian_text():
    text = """
    Консолидированная финансовая отчетность за 2024 год.
    Выручка 8 600 млн рублей. EBITDA 2 100 млн рублей.
    Чистая прибыль 850 млн рублей. Денежный поток от операционной деятельности 1 700 млн рублей.
    Свободный денежный поток -120 млн рублей. Капитальные затраты 1 820 млн рублей.
    Общий долг 5 000 млн рублей. Денежные средства и их эквиваленты 600 млн рублей.
    Процентные расходы 420 млн рублей. Итого активы 11 000 млн рублей. Итого капитал 3 900 млн рублей.
    В разделе рисков раскрыты санкции, валютный риск, риск ликвидности и ковенант.
    """
    features = extract_financial_report_features(text)
    assert features["report_revenue"] == 8_600_000_000
    assert features["report_ebitda"] == 2_100_000_000
    assert features["report_net_debt_to_ebitda"] > 2.0
    assert features["report_sanctions_flag"] == 1.0
    assert features["report_covenant_flag"] == 1.0
    assert features["report_financial_pressure"] > 0


def test_build_report_features_and_pit_merge_no_future_leakage():
    registry = pd.DataFrame(
        [
            {
                "ticker": "SBER",
                "company_name": "Sber",
                "report_period_end": "2023-12-31",
                "publish_date": "2024-03-01",
                "report_type": "ifrs",
                "report_text": "Выручка 1000 млн рублей. EBITDA 300 млн рублей. Общий долг 500 млн рублей. Денежные средства и их эквиваленты 100 млн рублей. Процентные расходы 50 млн рублей.",
            },
            {
                "ticker": "SBER",
                "company_name": "Sber",
                "report_period_end": "2024-12-31",
                "publish_date": "2025-03-01",
                "report_type": "ifrs",
                "report_text": "Выручка 2000 млн рублей. EBITDA 500 млн рублей. Общий долг 1000 млн рублей. Денежные средства и их эквиваленты 100 млн рублей. Процентные расходы 70 млн рублей. санкции.",
            },
        ]
    )
    features = build_financial_report_features(registry, include_evidence=False)
    assert set(features["parse_status"]) == {"parsed_embedded_text"}
    panel = pd.DataFrame(
        {
            "decision_date": pd.to_datetime(["2024-02-29", "2024-03-31", "2025-02-28", "2025-03-31"]),
            "ticker": ["SBER"] * 4,
            "sector": ["banks"] * 4,
        }
    )
    merged = merge_financial_report_features_pit(panel, features)
    assert merged.loc[0, "report_available"] == 0.0
    assert merged.loc[1, "report_revenue"] == 1_000_000_000
    assert merged.loc[2, "report_revenue"] == 1_000_000_000
    assert merged.loc[3, "report_revenue"] == 2_000_000_000
    assert merged.loc[3, "report_sanctions_flag"] == 1.0
