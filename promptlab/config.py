"""PromptLab 配置与测试集/提示词加载模块。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

# 项目根目录：promptlab/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CASES_DIR = PROJECT_ROOT / "cases"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RUNS_DIR = PROJECT_ROOT / "runs"

# 内置 IT 工单分类枚举与优先级枚举
ALLOWED_CATEGORIES: tuple[str, ...] = (
    "网络故障",
    "账号权限",
    "硬件设备",
    "软件应用",
    "其他",
)
ALLOWED_PRIORITIES: tuple[str, ...] = ("高", "中", "低")

# Magpie 网关真实模型列表（通过 GET http://127.0.0.1:3425/v1/models 获取）
DEFAULT_BASE_URL = "http://127.0.0.1:3425/v1"
DEFAULT_MODELS: list[str] = [
    "amd-tokenlab/Gemma-4-31B-IT",
    "amd-tokenlab/DeepSeek-V4-Flash",
    "amd-tokenlab/Qwen3.8-27B",
]


class CaseExpect(BaseModel):
    """单条测试用例的期望输出约束。"""

    category: str
    keywords: list[str] = Field(default_factory=list)


class TicketCase(BaseModel):
    """单条 IT 工单测试用例。"""

    id: str
    input: str
    expect: CaseExpect


class PromptVersion(BaseModel):
    """单个提示词版本定义。"""

    version: str
    description: str = ""
    system: str
    user_template: str = "{input}"

    def render_messages(self, ticket_input: str) -> list[dict[str, str]]:
        """渲染发送给 Chat Completions 接口的初始消息列表。"""
        return [
            {"role": "system", "content": self.system.strip()},
            {
                "role": "user",
                "content": self.user_template.format(input=ticket_input.strip()),
            },
        ]


class RunConfig(BaseModel):
    """执行引擎运行配置。"""

    base_url: str = DEFAULT_BASE_URL
    # 本地 Magpie 网关默认无需密钥；如需鉴权可通过环境变量传入
    api_key: str = Field(default_factory=lambda: os.getenv("MAGPIE_API_KEY", ""))
    models: list[str] = Field(default_factory=lambda: list(DEFAULT_MODELS))
    concurrency: int = Field(default=3, ge=1)
    timeout_seconds: float = Field(default=60.0, gt=0)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    # 两个独立重试预算：不合并计数
    network_max_retries: int = Field(default=3, ge=0)
    validation_max_retries: int = Field(default=2, ge=0)
    runs_dir: Path = RUNS_DIR


def load_cases(
    path: Path | None = None,
    limit: int | None = None,
    case_ids: list[str] | None = None,
) -> list[TicketCase]:
    """从 YAML 文件加载测试用例列表，支持按用例 ID 或数量上限过滤。"""
    target = path or (CASES_DIR / "tickets.yaml")
    raw: Any = yaml.safe_load(target.read_text(encoding="utf-8"))
    if isinstance(raw, dict):
        items = raw.get("cases", [])
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError(f"无法解析测试用例文件格式：{target}")

    cases = [TicketCase.model_validate(item) for item in items]
    if case_ids:
        wanted = set(case_ids)
        cases = [c for c in cases if c.id in wanted]
    if limit is not None:
        return cases[:limit]
    return cases


def load_prompt(path: Path) -> PromptVersion:
    """加载单个提示词版本文件（支持 .yaml/.yml）。"""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"提示词文件格式错误：{path}")
    return PromptVersion(
        version=str(data.get("version") or path.stem),
        description=str(data.get("description") or ""),
        system=str(data["system"]),
        user_template=str(data.get("user_template") or "{input}"),
    )


def load_prompts(
    prompts_dir: Path | None = None,
    versions: list[str] | None = None,
) -> list[PromptVersion]:
    """加载指定目录下的提示词版本列表，可按版本名过滤。"""
    directory = prompts_dir or PROMPTS_DIR
    files = sorted(
        list(directory.glob("*.yaml")) + list(directory.glob("*.yml")),
        key=lambda p: p.name,
    )
    loaded = [load_prompt(f) for f in files]
    if versions:
        wanted = set(versions)
        loaded = [p for p in loaded if p.version in wanted]
    return loaded
