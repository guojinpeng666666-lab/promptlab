"""输出校验器链模块。

按顺序执行：
1. json_schema：JSON 可解析且符合 Pydantic Schema（category、priority、summary）
2. category_enum：category 在允许的五类枚举内
3. summary_keywords：summary 包含测试用例期望的全部关键词
4. category_expect：category 与测试用例期望分类一致（不触发自纠重试，避免泄漏标准答案）
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from .config import ALLOWED_CATEGORIES, TicketCase


class TicketOutput(BaseModel):
    """模型输出工单结构的 Pydantic Schema。"""

    category: str = Field(..., min_length=1)
    priority: Literal["高", "中", "低"]
    summary: str = Field(..., min_length=1)

    @field_validator("category", "summary")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        """确保字段去除首尾空白后非空。"""
        stripped = v.strip()
        if not stripped:
            raise ValueError("字段不能为空白字符串")
        return stripped


class ValidationResult(BaseModel):
    """校验链单次执行结果。"""

    passed: bool
    failed_step: str | None = None
    reason: str | None = None
    parsed_output: dict[str, str] | None = None
    # True 表示格式/枚举/关键词等规范类错误，可追加失败原因让模型自纠；False 表示分类准确率错误
    retryable: bool = False


def validate_output(raw_text: str, case: TicketCase) -> ValidationResult:
    """按校验器链依次校验模型原始输出，任一步失败即返回 fail。"""
    text = (raw_text or "").strip()
    if not text:
        return ValidationResult(
            passed=False,
            failed_step="json_schema",
            reason="模型输出为空",
            retryable=True,
        )

    # ① JSON 可解析且符合 Pydantic Schema
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        return ValidationResult(
            passed=False,
            failed_step="json_schema",
            reason=f"JSON 解析失败：{exc.msg}（请仅输出纯 JSON，不要包含 Markdown 代码块或额外文本）",
            retryable=True,
        )

    if not isinstance(payload, dict):
        return ValidationResult(
            passed=False,
            failed_step="json_schema",
            reason="JSON 顶层必须是对象（dict），包含 category、priority、summary 字段",
            retryable=True,
        )

    try:
        parsed = TicketOutput.model_validate(payload)
    except ValidationError as exc:
        err_msgs = "; ".join(
            f"{'.'.join(str(x) for x in e['loc'])}: {e['msg']}" for e in exc.errors()
        )
        return ValidationResult(
            passed=False,
            failed_step="json_schema",
            reason=f"Schema 校验失败：{err_msgs}",
            retryable=True,
        )

    parsed_dict = parsed.model_dump()

    # ② category 在枚举内
    if parsed.category not in ALLOWED_CATEGORIES:
        allowed_str = "、".join(ALLOWED_CATEGORIES)
        return ValidationResult(
            passed=False,
            failed_step="category_enum",
            reason=f"category 值 '{parsed.category}' 不在允许枚举内（必须是：{allowed_str}）",
            parsed_output=parsed_dict,
            retryable=True,
        )

    # ③ summary 含用例要求关键词（不区分大小写）
    summary_lower = parsed.summary.lower()
    missing_keywords = [
        kw for kw in case.expect.keywords if kw.lower() not in summary_lower
    ]
    if missing_keywords:
        return ValidationResult(
            passed=False,
            failed_step="summary_keywords",
            reason=f"summary 缺少必含关键词：{', '.join(missing_keywords)}",
            parsed_output=parsed_dict,
            retryable=True,
        )

    # ④ 期望分类一致性校验（不可重试，避免把标准答案喂给模型）
    if parsed.category != case.expect.category:
        return ValidationResult(
            passed=False,
            failed_step="category_expect",
            reason=f"期望分类为「{case.expect.category}」，实际输出「{parsed.category}」",
            parsed_output=parsed_dict,
            retryable=False,
        )

    return ValidationResult(
        passed=True,
        parsed_output=parsed_dict,
        retryable=False,
    )
