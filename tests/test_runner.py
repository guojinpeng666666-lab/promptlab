"""runner.py 单元测试（使用 httpx.MockTransport，不消耗真实网关额度）。"""

import asyncio
import json
from pathlib import Path

import httpx

from promptlab.config import CaseExpect, PromptVersion, RunConfig, TicketCase
from promptlab.runner import run_matrix


def _sample_prompt(version: str = "v1") -> PromptVersion:
    return PromptVersion(version=version, system="只输出JSON", user_template="{input}")


def _sample_case(case_id: str = "T001") -> TicketCase:
    return TicketCase(
        id=case_id,
        input="VPN 无法连接，认证超时",
        expect=CaseExpect(category="网络故障", keywords=["VPN"]),
    )


def _ok_resp(content: str, p_tok: int = 10, c_tok: int = 20) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "choices": [{"message": {"role": "assistant", "content": content}}],
            "usage": {
                "prompt_tokens": p_tok,
                "completion_tokens": c_tok,
                "total_tokens": p_tok + c_tok,
            },
        },
    )


def test_independent_retry_budgets_do_not_merge(tmp_path: Path):
    """网络错误消耗满 3 次预算后，校验错误仍保有独立 2 次自纠预算。"""
    call_count = 0
    seen_last_user_msgs: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        body = json.loads(request.content.decode("utf-8"))
        seen_last_user_msgs.append(body["messages"][-1]["content"])

        if call_count == 1:
            raise httpx.ConnectError("连接失败", request=request)
        if call_count == 2:
            return httpx.Response(502, text="bad gateway")
        if call_count == 3:
            raise httpx.ReadTimeout("请求超时", request=request)
        if call_count == 4:
            # 第一次拿到 200，但故意输出非 JSON，触发校验自纠预算
            return _ok_resp("这不是JSON", p_tok=10, c_tok=5)
        # 第 5 次调用是校验自纠重试：应该收到追加的失败原因
        return _ok_resp(
            '{"category": "网络故障", "priority": "高", "summary": "VPN 连接认证超时"}',
            p_tok=20,
            c_tok=15,
        )

    cfg = RunConfig(
        models=["test-model"],
        concurrency=2,
        network_max_retries=3,
        validation_max_retries=2,
        runs_dir=tmp_path,
    )

    async def _run():
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            return await run_matrix(
                cfg,
                [_sample_prompt()],
                [_sample_case()],
                client=client,
                retry_backoff_base=0.0,
            )

    report, out_path = asyncio.run(_run())
    assert out_path.exists()
    assert call_count == 5
    res = report.results[0]
    assert res.passed is True
    assert res.network_retries == 3
    assert res.validation_retries == 1
    assert res.total_tokens == 50  # (10+5) + (20+15)
    assert "json_schema" in seen_last_user_msgs[-1]


def test_network_budget_exhausted_counted_separately(tmp_path: Path):
    """网络错误耗尽 3 次预算后计入 network_error_count，不影响 validation_error_count。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="service unavailable")

    cfg = RunConfig(
        models=["test-model"],
        network_max_retries=3,
        validation_max_retries=2,
        runs_dir=tmp_path,
    )

    async def _run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_matrix(
                cfg,
                [_sample_prompt()],
                [_sample_case()],
                client=client,
                retry_backoff_base=0.0,
            )

    report, _ = asyncio.run(_run())
    assert report.totals.network_error_count == 1
    assert report.totals.validation_error_count == 0
    assert report.totals.total_network_retries == 3
    assert report.totals.total_validation_retries == 0
    assert report.results[0].failure_type == "network_error"


def test_validation_budget_exhausted_counted_separately(tmp_path: Path):
    """输出校验连续失败耗尽 2 次预算后计入 validation_error_count。"""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        call_count += 1
        # 始终缺少必含关键词 VPN
        return _ok_resp('{"category": "网络故障", "priority": "高", "summary": "内网连接异常"}')

    cfg = RunConfig(
        models=["test-model"],
        network_max_retries=3,
        validation_max_retries=2,
        runs_dir=tmp_path,
    )

    async def _run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_matrix(
                cfg,
                [_sample_prompt()],
                [_sample_case()],
                client=client,
                retry_backoff_base=0.0,
            )

    report, _ = asyncio.run(_run())
    assert call_count == 3  # 1 次初试 + 2 次自纠重试
    assert report.totals.validation_error_count == 1
    assert report.totals.network_error_count == 0
    assert report.totals.total_validation_retries == 2
    assert report.totals.total_network_retries == 0
    assert report.results[0].failed_step == "summary_keywords"


def test_matrix_cells_and_json_persistence(tmp_path: Path):
    """验证 2 模型 × 2 版本 × 2 用例矩阵聚合与落盘结构。"""

    def handler(request: httpx.Request) -> httpx.Response:
        return _ok_resp(
            '{"category": "网络故障", "priority": "高", "summary": "VPN 异常"}',
            p_tok=10,
            c_tok=10,
        )

    cfg = RunConfig(models=["m1", "m2"], concurrency=3, runs_dir=tmp_path)
    prompts = [_sample_prompt("v1"), _sample_prompt("v2")]
    cases = [_sample_case("T001"), _sample_case("T002")]

    async def _run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await run_matrix(
                cfg,
                prompts,
                cases,
                client=client,
                retry_backoff_base=0.0,
            )

    report, out_path = asyncio.run(_run())
    assert len(report.matrix) == 4
    assert report.totals.total_tasks == 8
    assert report.totals.passed == 8
    assert report.totals.total_tokens == 160
    saved = json.loads(out_path.read_text(encoding="utf-8"))
    assert len(saved["results"]) == 8


def test_load_cases_filter_by_case_ids():
    """验证按 case_ids 精确过滤测试用例。"""
    from promptlab.config import load_cases

    filtered = load_cases(case_ids=["T003", "T006"])
    assert [c.id for c in filtered] == ["T003", "T006"]
