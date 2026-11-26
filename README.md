# PromptLab（提示词测试与评估平台）

用于批量评测多模型、多提示词版本在结构化任务上的真实表现：自动并发调用 OpenAI 兼容接口、执行校验器链校验输出、记录延迟与 Token 消耗，并生成静态 HTML 通过率对比报告。

## 目录结构

```text
promptlab/
  cases/              # 测试集 YAML（如 tickets.yaml）
  prompts/            # 提示词版本文件（如 v1_base.yaml、v2_strict.yaml）
  promptlab/
    __init__.py
    config.py         # 配置、模型列表与 YAML 加载器
    validators.py     # 四步校验器链
    runner.py         # 异步矩阵执行引擎与双预算重试
    report.py         # Jinja2 静态 HTML 报告生成器
  runs/               # 每次运行的 JSON 原始结果与 HTML 报告
  tests/              # pytest 离线单元测试（不消耗网关额度）
  README.md
```

## 运行环境与依赖

- Python 3.11+（本机已验证 Python 3.13）
- 运行依赖：`httpx`、`pydantic`、`pyyaml`、`jinja2`
- 测试依赖：`pytest`
- 默认对接本机 Magpie 网关：`http://127.0.0.1:3425/v1`（若需鉴权，设置环境变量 `MAGPIE_API_KEY`，不要把密钥写入代码或配置文件）

## 常用运行命令

在 `promptlab/` 目录下执行：

```powershell
# 1. 运行离线单元测试（Mock HTTP，不花额度）
python -m pytest -v

# 2. 小批量冒烟测试（1 模型 × 1 版本 × 前 5 条用例）
python -m promptlab.runner --models amd-tokenlab/Gemma-4-31B-IT --prompts v1_base --limit 5 --concurrency 3

# 3. 指定模型、提示词版本与用例 ID 运行对比矩阵
python -m promptlab.runner --models amd-tokenlab/Gemma-4-31B-IT amd-tokenlab/DeepSeek-V4-Flash --prompts v1_base v2_strict --case-ids T003 T006 --concurrency 3

# 4. 运行默认全矩阵（注意额度消耗：3 模型 × 2 版本 × 15 用例 = 90+ 次调用）
python -m promptlab.runner

# 5. 根据运行结果 JSON 生成静态 HTML 报告（或用 --latest 自动选最新结果）
python -m promptlab.report runs/20261125_164848.json
python -m promptlab.report --latest
```

## 核心设计说明

### 1. 双独立重试预算（不合并计数）

`runner.py` 对每条 `(模型, 提示词版本, 用例)` 任务维护两个完全隔离的重试计数器，并在日志和 `runs/<时间戳>.json` 中分开统计：

- **网络/服务类错误预算（默认 3 次）**：针对连接失败、请求超时、`429` 限流及 `5xx` 服务端错误，仅消耗网络重试预算并做指数退避，不修改对话上下文，也不占用校验自纠预算。
- **输出校验失败预算（默认 2 次）**：当网关返回 `HTTP 200` 但输出未通过格式/约束校验时，把具体的失败步骤与原因追加到对话消息历史（`assistant` 原输出 + `user` 纠错提示）让模型自纠，仅消耗校验自纠预算。

### 2. 校验器链与防答案泄漏设计

`validators.py` 按顺序执行 4 个校验步骤，任一步失败即判 `fail`：

1. `json_schema`：校验输出是否为纯 JSON 对象且符合 `TicketOutput` Pydantic Schema（`category`、`priority ∈ {"高","中","低"}`、`summary` 非空）。
2. `category_enum`：校验 `category` 是否在允许的五类枚举内（`网络故障`、`账号权限`、`硬件设备`、`软件应用`、`其他`）。
3. `summary_keywords`：校验 `summary` 是否包含用例 `expect.keywords` 要求的关键词（不区分大小写）。
4. `category_expect`：校验 `category` 是否等于用例标准答案 `expect.category`。

**防答案泄漏**：前 3 步属于输出格式与信息完整度约束，标记为 `retryable=True`，允许触发自纠重试；第 4 步 `category_expect` 标记为 `retryable=False`，一旦分类错误直接判 `fail` 结束该用例，**不会把期望分类回传给模型重试**，从而保证测试集准确率统计的真实性。
