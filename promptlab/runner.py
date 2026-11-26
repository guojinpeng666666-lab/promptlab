"""异步矩阵执行引擎。

核心设计：
- 按「模型 × 提示词版本 × 测试用例」构造任务矩阵，使用 asyncio.Semaphore 控制并发（默认 3）。
- 维护两个完全独立的重试预算，严禁合并：
  1. 网络/服务类错误（超时、连接失败、5xx、429）：默认预算 3 次；
  2. 输出校验失败（JSON/Schema/枚举/关键词）：默认预算 2 次，并将失败原因追加到对话历史中让模型自纠。
- 每次运行将完整结果（原始输出、校验结果、延迟、token 用量、两类失败与重试计数）保存至 runs/<时间戳>.json。
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime
import json
import logging
from pathlib import Path
import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from .config import (
    PromptVersion,
    RunConfig,
    TicketCase,
    load_cases,
    load_prompts,
)
from .validators import validate_output

logger = logging.getLogger("promptlab.runner")


class AttemptTrace(BaseModel):
    """单次请求尝试记录。"""

    attempt_index: int
    outcome: Literal["pass", "network_error", "validation_error"]
    status_code: int | None = None
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    failed_step: str | None = None
    error_message: str | None = None
    raw_output: str | None = None


class CaseRunResult(BaseModel):
    """单个（模型 × 提示词版本 × 用例）的最终执行结果。"""

    model: str
    prompt_version: str
    case_id: str
    passed: bool
    failure_type: Literal["none", "network_error", "validation_error"] = "none"
    failed_step: str | None = None
    failure_reason: str | None = None
    raw_output: str | None = None
    parsed_output: dict[str, str] | None = None
    network_retries: int = 0
    validation_retries: int = 0
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    attempts: list[AttemptTrace] = Field(default_factory=list)


class MatrixCellSummary(BaseModel):
    """矩阵单元格（单个模型 × 单个提示词版本）汇总指标。"""

    model: str
    prompt_version: str
    total: int
    passed: int
    failed: int
    pass_rate: float
    avg_latency_ms: float
    total_tokens: int
    network_errors: int
    validation_errors: int
    network_retries: int
    validation_retries: int


class RunTotals(BaseModel):
    """整次运行全局汇总（两类失败分开计数）。"""

    total_tasks: int
    passed: int
    failed: int
    pass_rate: float
    network_error_count: int
    validation_error_count: int
    total_network_retries: int
    total_validation_retries: int
    avg_latency_ms: float
    total_tokens: int


class RunReportData(BaseModel):
    """落盘到 runs/<时间戳>.json 的完整运行数据结构。"""

    run_id: str
    timestamp: str
    base_url: str
    models: list[str]
    prompt_versions: list[str]
    case_ids: list[str]
    concurrency: int
    network_max_retries: int
    validation_max_retries: int
    totals: RunTotals
    matrix: list[MatrixCellSummary]
    results: list[CaseRunResult]


def _extract_content_and_usage(data: dict[str, Any]) -> tuple[str, int, int, int]:
    """从 OpenAI 兼容 /chat/completions 响应中提取正文与 token 用量。"""
    choices = data.get("choices") or []
    content = ""
    if choices and isinstance(choices[0], dict):
        msg = choices[0].get("message") or {}
        raw_content = msg.get("content")
        if isinstance(raw_content, str):
            content = raw_content
        elif isinstance(raw_content, list):
            # 兼容少数网关返回多段 content block 的情况
            content = "".join(
                part.get("text", "") for part in raw_content if isinstance(part, dict)
            )

    usage = data.get("usage") or {}
    prompt_tokens = int(usage.get("prompt_tokens") or 0)
    completion_tokens = int(usage.get("completion_tokens") or 0)
    total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
    return content, prompt_tokens, completion_tokens, total_tokens


async def _run_single_case(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    model: str,
    prompt: PromptVersion,
    case: TicketCase,
    config: RunConfig,
    retry_backoff_base: float = 0.5,
) -> CaseRunResult:
    """执行单个（模型 × 版本 × 用例），严格隔离网络重试与校验重试两个预算。"""
    messages = prompt.render_messages(case.input)
    network_retries_used = 0
    validation_retries_used = 0
    total_latency_ms = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    attempts: list[AttemptTrace] = []
    last_raw_output: str | None = None

    endpoint = f"{config.base_url.rstrip('/')}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"

    while True:
        attempt_idx = len(attempts) + 1
        payload = {
            "model": model,
            "messages": messages,
            "temperature": config.temperature,
        }

        t0 = time.perf_counter()
        response: httpx.Response | None = None
        network_err_msg: str | None = None
        status_code: int | None = None
        retryable_network_err = False

        async with semaphore:
            try:
                response = await client.post(
                    endpoint,
                    headers=headers,
                    json=payload,
                    timeout=config.timeout_seconds,
                )
                status_code = response.status_code
                if status_code >= 500 or status_code == 429:
                    retryable_network_err = True
                    network_err_msg = f"HTTP {status_code}: {response.text[:200]}"
                elif status_code >= 400:
                    retryable_network_err = False
                    network_err_msg = f"HTTP {status_code}: {response.text[:300]}"
            except (httpx.TimeoutException, httpx.ConnectError, httpx.TransportError) as exc:
                retryable_network_err = True
                network_err_msg = f"{type(exc).__name__}: {exc}"

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        total_latency_ms += elapsed_ms

        # 分支 A：网络/服务类错误 -> 仅消耗 network_max_retries 预算
        if network_err_msg is not None:
            attempts.append(
                AttemptTrace(
                    attempt_index=attempt_idx,
                    outcome="network_error",
                    status_code=status_code,
                    latency_ms=round(elapsed_ms, 2),
                    failed_step="api_call",
                    error_message=network_err_msg,
                )
            )
            if retryable_network_err and network_retries_used < config.network_max_retries:
                network_retries_used += 1
                logger.warning(
                    "[网络重试 %d/%d] model=%s version=%s case=%s err=%s",
                    network_retries_used,
                    config.network_max_retries,
                    model,
                    prompt.version,
                    case.id,
                    network_err_msg,
                )
                if retry_backoff_base > 0:
                    await asyncio.sleep(
                        min(retry_backoff_base * (2 ** (network_retries_used - 1)), 4.0)
                    )
                continue

            logger.error(
                "[网络/服务失败] model=%s version=%s case=%s network_retries=%d err=%s",
                model,
                prompt.version,
                case.id,
                network_retries_used,
                network_err_msg,
            )
            return CaseRunResult(
                model=model,
                prompt_version=prompt.version,
                case_id=case.id,
                passed=False,
                failure_type="network_error",
                failed_step="api_call",
                failure_reason=network_err_msg,
                raw_output=last_raw_output,
                network_retries=network_retries_used,
                validation_retries=validation_retries_used,
                latency_ms=round(total_latency_ms, 2),
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
                attempts=attempts,
            )

        # HTTP 200：解析返回内容与 token 用量
        assert response is not None
        try:
            resp_json = response.json()
        except ValueError as exc:
            err_msg = f"响应体非合法 JSON：{exc}"
            attempts.append(
                AttemptTrace(
                    attempt_index=attempt_idx,
                    outcome="network_error",
                    status_code=status_code,
                    latency_ms=round(elapsed_ms, 2),
                    failed_step="api_response_json",
                    error_message=err_msg,
                    raw_output=response.text[:500],
                )
            )
            if network_retries_used < config.network_max_retries:
                network_retries_used += 1
                logger.warning(
                    "[网络重试 %d/%d] model=%s version=%s case=%s err=%s",
                    network_retries_used,
                    config.network_max_retries,
                    model,
                    prompt.version,
                    case.id,
                    err_msg,
                )
                continue
            return CaseRunResult(
                model=model,
                prompt_version=prompt.version,
                case_id=case.id,
                passed=False,
                failure_type="network_error",
                failed_step="api_response_json",
                failure_reason=err_msg,
                raw_output=response.text[:500],
                network_retries=network_retries_used,
                validation_retries=validation_retries_used,
                latency_ms=round(total_latency_ms, 2),
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
                attempts=attempts,
            )

        raw_output, p_tok, c_tok, t_tok = _extract_content_and_usage(resp_json)
        last_raw_output = raw_output
        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        total_tokens += t_tok

        # 分支 B：运行校验器链
        v_res = validate_output(raw_output, case)
        if v_res.passed:
            attempts.append(
                AttemptTrace(
                    attempt_index=attempt_idx,
                    outcome="pass",
                    status_code=status_code,
                    latency_ms=round(elapsed_ms, 2),
                    prompt_tokens=p_tok,
                    completion_tokens=c_tok,
                    total_tokens=t_tok,
                    raw_output=raw_output,
                )
            )
            return CaseRunResult(
                model=model,
                prompt_version=prompt.version,
                case_id=case.id,
                passed=True,
                failure_type="none",
                raw_output=raw_output,
                parsed_output=v_res.parsed_output,
                network_retries=network_retries_used,
                validation_retries=validation_retries_used,
                latency_ms=round(total_latency_ms, 2),
                prompt_tokens=total_prompt_tokens,
                completion_tokens=total_completion_tokens,
                total_tokens=total_tokens,
                attempts=attempts,
            )

        attempts.append(
            AttemptTrace(
                attempt_index=attempt_idx,
                outcome="validation_error",
                status_code=status_code,
                latency_ms=round(elapsed_ms, 2),
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=t_tok,
                failed_step=v_res.failed_step,
                error_message=v_res.reason,
                raw_output=raw_output,
            )
        )

        # 校验失败且允许自纠 -> 仅消耗 validation_max_retries 预算，并把失败原因追加到对话
        if v_res.retryable and validation_retries_used < config.validation_max_retries:
            validation_retries_used += 1
            logger.warning(
                "[校验自纠重试 %d/%d] model=%s version=%s case=%s step=%s reason=%s",
                validation_retries_used,
                config.validation_max_retries,
                model,
                prompt.version,
                case.id,
                v_res.failed_step,
                v_res.reason,
            )
            messages = [
                *messages,
                {"role": "assistant", "content": raw_output},
                {
                    "role": "user",
                    "content": (
                        f"你的上一次输出未通过校验（失败步骤：{v_res.failed_step}，"
                        f"原因：{v_res.reason}）。请严格按格式和内容要求修正，"
                        "并重新仅输出一个合法的纯 JSON 对象。"
                    ),
                },
            ]
            continue

        logger.warning(
            "[校验失败] model=%s version=%s case=%s validation_retries=%d step=%s reason=%s",
            model,
            prompt.version,
            case.id,
            validation_retries_used,
            v_res.failed_step,
            v_res.reason,
        )
        return CaseRunResult(
            model=model,
            prompt_version=prompt.version,
            case_id=case.id,
            passed=False,
            failure_type="validation_error",
            failed_step=v_res.failed_step,
            failure_reason=v_res.reason,
            raw_output=raw_output,
            parsed_output=v_res.parsed_output,
            network_retries=network_retries_used,
            validation_retries=validation_retries_used,
            latency_ms=round(total_latency_ms, 2),
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            total_tokens=total_tokens,
            attempts=attempts,
        )


def _build_matrix_and_totals(
    models: list[str],
    prompts: list[PromptVersion],
    results: list[CaseRunResult],
) -> tuple[list[MatrixCellSummary], RunTotals]:
    """根据逐条结果聚合「模型 × 提示词版本」矩阵与全局统计。"""
    by_cell: dict[tuple[str, str], list[CaseRunResult]] = {}
    for r in results:
        by_cell.setdefault((r.model, r.prompt_version), []).append(r)

    matrix: list[MatrixCellSummary] = []
    for model in models:
        for prompt in prompts:
            cell_items = by_cell.get((model, prompt.version), [])
            total = len(cell_items)
            passed = sum(1 for x in cell_items if x.passed)
            failed = total - passed
            pass_rate = round((passed / total * 100.0) if total else 0.0, 2)
            avg_lat = round(
                (sum(x.latency_ms for x in cell_items) / total) if total else 0.0, 2
            )
            tok = sum(x.total_tokens for x in cell_items)
            net_errs = sum(1 for x in cell_items if x.failure_type == "network_error")
            val_errs = sum(1 for x in cell_items if x.failure_type == "validation_error")
            net_retries = sum(x.network_retries for x in cell_items)
            val_retries = sum(x.validation_retries for x in cell_items)
            matrix.append(
                MatrixCellSummary(
                    model=model,
                    prompt_version=prompt.version,
                    total=total,
                    passed=passed,
                    failed=failed,
                    pass_rate=pass_rate,
                    avg_latency_ms=avg_lat,
                    total_tokens=tok,
                    network_errors=net_errs,
                    validation_errors=val_errs,
                    network_retries=net_retries,
                    validation_retries=val_retries,
                )
            )

    total_tasks = len(results)
    total_passed = sum(1 for x in results if x.passed)
    total_failed = total_tasks - total_passed
    totals = RunTotals(
        total_tasks=total_tasks,
        passed=total_passed,
        failed=total_failed,
        pass_rate=round((total_passed / total_tasks * 100.0) if total_tasks else 0.0, 2),
        network_error_count=sum(1 for x in results if x.failure_type == "network_error"),
        validation_error_count=sum(
            1 for x in results if x.failure_type == "validation_error"
        ),
        total_network_retries=sum(x.network_retries for x in results),
        total_validation_retries=sum(x.validation_retries for x in results),
        avg_latency_ms=round(
            (sum(x.latency_ms for x in results) / total_tasks) if total_tasks else 0.0,
            2,
        ),
        total_tokens=sum(x.total_tokens for x in results),
    )
    return matrix, totals


def _allocate_run_path(runs_dir: Path, run_id: str) -> Path:
    """生成不冲突的 runs/<时间戳>.json 文件路径。"""
    runs_dir.mkdir(parents=True, exist_ok=True)
    candidate = runs_dir / f"{run_id}.json"
    if not candidate.exists():
        return candidate
    suffix = 1
    while True:
        alt = runs_dir / f"{run_id}_{suffix:02d}.json"
        if not alt.exists():
            return alt
        suffix += 1


async def run_matrix(
    config: RunConfig,
    prompts: list[PromptVersion],
    cases: list[TicketCase],
    *,
    client: httpx.AsyncClient | None = None,
    retry_backoff_base: float = 0.5,
) -> tuple[RunReportData, Path]:
    """执行「模型 × 提示词版本 × 用例」全矩阵评估并保存结果 JSON。"""
    if not config.models:
        raise ValueError("models 不能为空")
    if not prompts:
        raise ValueError("prompts 不能为空")
    if not cases:
        raise ValueError("cases 不能为空")

    semaphore = asyncio.Semaphore(config.concurrency)
    own_client = client is None
    active_client = client or httpx.AsyncClient()

    try:
        tasks = [
            _run_single_case(
                client=active_client,
                semaphore=semaphore,
                model=model,
                prompt=prompt,
                case=case,
                config=config,
                retry_backoff_base=retry_backoff_base,
            )
            for model in config.models
            for prompt in prompts
            for case in cases
        ]
        results = list(await asyncio.gather(*tasks))
    finally:
        if own_client:
            await active_client.aclose()

    now = datetime.now()
    run_id = now.strftime("%Y%m%d_%H%M%S")
    matrix, totals = _build_matrix_and_totals(config.models, prompts, results)

    report = RunReportData(
        run_id=run_id,
        timestamp=now.isoformat(timespec="seconds"),
        base_url=config.base_url,
        models=list(config.models),
        prompt_versions=[p.version for p in prompts],
        case_ids=[c.id for c in cases],
        concurrency=config.concurrency,
        network_max_retries=config.network_max_retries,
        validation_max_retries=config.validation_max_retries,
        totals=totals,
        matrix=matrix,
        results=results,
    )

    out_path = _allocate_run_path(config.runs_dir, run_id)
    out_path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "运行完成：总任务=%d 通过=%d 通过率=%.2f%% | 网络/服务失败=%d(重试=%d) | 校验失败=%d(自纠重试=%d) | 总Token=%d | 结果=%s",
        totals.total_tasks,
        totals.passed,
        totals.pass_rate,
        totals.network_error_count,
        totals.total_network_retries,
        totals.validation_error_count,
        totals.total_validation_retries,
        totals.total_tokens,
        out_path,
    )
    return report, out_path


def main(argv: list[str] | None = None) -> int:
    """命令行入口：支持指定模型、提示词版本、用例文件与数量上限。"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="PromptLab 批量执行引擎")
    parser.add_argument("--models", nargs="+", help="指定模型列表，不传则使用默认模型")
    parser.add_argument("--prompts", nargs="+", help="指定提示词版本名，不传则加载全部")
    parser.add_argument("--cases", type=Path, help="测试集 YAML 路径，默认 cases/tickets.yaml")
    parser.add_argument("--case-ids", nargs="+", help="仅运行指定 ID 的测试用例")
    parser.add_argument("--limit", type=int, help="仅运行前 N 条用例（用于小批量冒烟）")
    parser.add_argument("--concurrency", type=int, default=3, help="并发度（默认 3）")
    parser.add_argument("--base-url", type=str, help="网关 base_url，覆盖默认配置")
    args = parser.parse_args(argv)

    cfg = RunConfig(concurrency=args.concurrency)
    if args.models:
        cfg.models = args.models
    if args.base_url:
        cfg.base_url = args.base_url

    cases = load_cases(path=args.cases, limit=args.limit, case_ids=args.case_ids)
    prompts = load_prompts(versions=args.prompts)

    report, out_path = asyncio.run(run_matrix(cfg, prompts, cases))
    print(f"结果已保存: {out_path}")
    print(
        f"总计 {report.totals.total_tasks} | 通过 {report.totals.passed} ({report.totals.pass_rate}%) | "
        f"网络失败 {report.totals.network_error_count} (重试 {report.totals.total_network_retries}) | "
        f"校验失败 {report.totals.validation_error_count} (自纠 {report.totals.total_validation_retries}) | "
        f"平均延迟 {report.totals.avg_latency_ms} ms | 总 Token {report.totals.total_tokens}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
