"""report.py 单元测试。"""

from pathlib import Path

from promptlab.report import format_matrix_text, generate_html_report
from promptlab.runner import (
    CaseRunResult,
    MatrixCellSummary,
    RunReportData,
    RunTotals,
)


def test_generate_html_report_escapes_and_lists_failures(tmp_path: Path):
    """验证报告包含矩阵指标、失败明细且对原始输出做 HTML 转义。"""
    sample = RunReportData(
        run_id="20261125_000000",
        timestamp="2026-11-25T00:00:00",
        base_url="http://127.0.0.1:3425/v1",
        models=["model-a", "model-b"],
        prompt_versions=["v1_base", "v2_strict"],
        case_ids=["T003", "T006"],
        concurrency=3,
        network_max_retries=3,
        validation_max_retries=2,
        totals=RunTotals(
            total_tasks=4,
            passed=3,
            failed=1,
            pass_rate=75.0,
            network_error_count=0,
            validation_error_count=1,
            total_network_retries=0,
            total_validation_retries=1,
            avg_latency_ms=1200.5,
            total_tokens=400,
        ),
        matrix=[
            MatrixCellSummary(
                model="model-a",
                prompt_version="v1_base",
                total=1,
                passed=1,
                failed=0,
                pass_rate=100.0,
                avg_latency_ms=1000.0,
                total_tokens=100,
                network_errors=0,
                validation_errors=0,
                network_retries=0,
                validation_retries=0,
            ),
            MatrixCellSummary(
                model="model-a",
                prompt_version="v2_strict",
                total=1,
                passed=1,
                failed=0,
                pass_rate=100.0,
                avg_latency_ms=1100.0,
                total_tokens=100,
                network_errors=0,
                validation_errors=0,
                network_retries=0,
                validation_retries=0,
            ),
            MatrixCellSummary(
                model="model-b",
                prompt_version="v1_base",
                total=1,
                passed=0,
                failed=1,
                pass_rate=0.0,
                avg_latency_ms=1500.0,
                total_tokens=100,
                network_errors=0,
                validation_errors=1,
                network_retries=0,
                validation_retries=1,
            ),
            MatrixCellSummary(
                model="model-b",
                prompt_version="v2_strict",
                total=1,
                passed=1,
                failed=0,
                pass_rate=100.0,
                avg_latency_ms=1202.0,
                total_tokens=100,
                network_errors=0,
                validation_errors=0,
                network_retries=0,
                validation_retries=0,
            ),
        ],
        results=[
            CaseRunResult(
                model="model-b",
                prompt_version="v1_base",
                case_id="T006",
                passed=False,
                failure_type="validation_error",
                failed_step="category_expect",
                failure_reason="期望分类为「账号权限」，实际输出「软件应用」",
                raw_output='<script>alert(1)</script>{"category":"软件应用"}',
                latency_ms=1500.0,
                total_tokens=100,
                validation_retries=1,
            ),
        ],
    )
    json_path = tmp_path / "run.json"
    json_path.write_text(sample.model_dump_json(indent=2), encoding="utf-8")

    loaded, html_path = generate_html_report(json_path)
    html = html_path.read_text(encoding="utf-8")
    assert html_path.exists()
    assert "model-a" in html
    assert "v2_strict" in html
    assert "T006" in html
    assert "期望分类为「账号权限」，实际输出「软件应用」" in html
    # Jinja2 autoescape 应转义 <script>
    assert "<script>alert(1)</script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html

    matrix_text = format_matrix_text(loaded)
    assert "model-b" in matrix_text
    assert "0.0% (0/1)" in matrix_text
