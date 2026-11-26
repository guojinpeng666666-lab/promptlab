"""validators.py 单元测试。"""

from promptlab.config import CaseExpect, TicketCase
from promptlab.validators import validate_output


def _make_case(category: str = "网络故障", keywords: list[str] | None = None) -> TicketCase:
    return TicketCase(
        id="T001",
        input="测试工单",
        expect=CaseExpect(
            category=category,
            keywords=keywords if keywords is not None else ["VPN"],
        ),
    )


def test_validate_pass():
    case = _make_case(category="网络故障", keywords=["VPN", "超时"])
    raw = '{"category": "网络故障", "priority": "高", "summary": "员工连接 VPN 出现超时问题"}'
    res = validate_output(raw, case)
    assert res.passed is True
    assert res.failed_step is None
    assert res.parsed_output == {
        "category": "网络故障",
        "priority": "高",
        "summary": "员工连接 VPN 出现超时问题",
    }


def test_validate_invalid_json_fence():
    case = _make_case()
    raw = '```json\n{"category": "网络故障", "priority": "高", "summary": "VPN 异常"}\n```'
    res = validate_output(raw, case)
    assert res.passed is False
    assert res.failed_step == "json_schema"
    assert res.retryable is True


def test_validate_invalid_priority_schema():
    case = _make_case()
    raw = '{"category": "网络故障", "priority": "紧急", "summary": "VPN 异常"}'
    res = validate_output(raw, case)
    assert res.passed is False
    assert res.failed_step == "json_schema"
    assert res.retryable is True


def test_validate_category_not_in_enum():
    case = _make_case()
    raw = '{"category": "安全漏洞", "priority": "高", "summary": "VPN 异常"}'
    res = validate_output(raw, case)
    assert res.passed is False
    assert res.failed_step == "category_enum"
    assert res.retryable is True


def test_validate_missing_keywords_case_insensitive():
    case = _make_case(keywords=["VPN", "GitLab"])
    # 小写 vpn 应该匹配，缺少 GitLab 应失败
    raw = '{"category": "网络故障", "priority": "中", "summary": "无法连接 vpn 隧道"}'
    res = validate_output(raw, case)
    assert res.passed is False
    assert res.failed_step == "summary_keywords"
    assert "GitLab" in (res.reason or "")
    assert res.retryable is True


def test_validate_category_mismatch_not_retryable():
    case = _make_case(category="账号权限", keywords=["报销系统"])
    raw = '{"category": "软件应用", "priority": "高", "summary": "报销系统提示 403 错误"}'
    res = validate_output(raw, case)
    assert res.passed is False
    assert res.failed_step == "category_expect"
    assert res.retryable is False
