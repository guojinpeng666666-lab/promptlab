"""静态 HTML 报告生成模块（基于 Jinja2）。"""

from __future__ import annotations

import argparse
from pathlib import Path

from jinja2 import Environment, select_autoescape

from .config import RUNS_DIR
from .runner import MatrixCellSummary, RunReportData

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PromptLab 评估报告 - {{ report.run_id }}</title>
  <style>
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Microsoft YaHei", sans-serif;
      color: #1f2937;
      background: #f9fafb;
      margin: 24px auto;
      max-width: 1200px;
      padding: 0 16px;
      line-height: 1.5;
    }
    h1, h2 { color: #111827; margin-bottom: 8px; }
    .meta {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 12px 16px;
      margin-bottom: 20px;
      font-size: 14px;
    }
    .meta-grid {
      display: flex;
      flex-wrap: wrap;
      gap: 16px 28px;
    }
    .tag {
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 12px;
      font-weight: 600;
    }
    .tag-ok { background: #dcfce7; color: #166534; }
    .tag-warn { background: #fef3c7; color: #92400e; }
    .tag-err { background: #fee2e2; color: #991b1b; }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      margin-bottom: 24px;
      font-size: 14px;
    }
    th, td {
      border: 1px solid #d1d5db;
      padding: 10px 12px;
      text-align: left;
      vertical-align: top;
    }
    th { background: #f3f4f6; font-weight: 600; }
    .cell-rate { font-size: 16px; font-weight: 700; }
    .cell-sub { font-size: 12px; color: #4b5563; margin-top: 4px; }
    pre.raw-output {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-all;
      font-family: Consolas, "Courier New", monospace;
      font-size: 12px;
      background: #f8fafc;
      padding: 6px 8px;
      border-radius: 4px;
      border: 1px solid #e2e8f0;
      max-width: 420px;
    }
    .empty-box {
      background: #ffffff;
      border: 1px solid #e5e7eb;
      padding: 16px;
      border-radius: 6px;
      color: #166534;
    }
  </style>
</head>
<body>
  <h1>PromptLab 评估报告</h1>

  <div class="meta">
    <div class="meta-grid">
      <div><strong>Run ID：</strong>{{ report.run_id }}</div>
      <div><strong>时间：</strong>{{ report.timestamp }}</div>
      <div><strong>网关：</strong>{{ report.base_url }}</div>
      <div><strong>用例数：</strong>{{ report.case_ids | length }}</div>
      <div><strong>总任务：</strong>{{ report.totals.total_tasks }}</div>
      <div><strong>总通过率：</strong>{{ "%.2f" | format(report.totals.pass_rate) }}% ({{ report.totals.passed }}/{{ report.totals.total_tasks }})</div>
      <div><strong>平均延迟：</strong>{{ "%.2f" | format(report.totals.avg_latency_ms) }} ms</div>
      <div><strong>总 Token：</strong>{{ report.totals.total_tokens }}</div>
      <div><strong>网络/服务失败：</strong>{{ report.totals.network_error_count }}（重试 {{ report.totals.total_network_retries }} 次）</div>
      <div><strong>输出校验失败：</strong>{{ report.totals.validation_error_count }}（自纠重试 {{ report.totals.total_validation_retries }} 次）</div>
    </div>
  </div>

  <h2>模型 × 提示词版本 通过率矩阵</h2>
  <table>
    <thead>
      <tr>
        <th>模型 / 提示词版本</th>
        {% for ver in report.prompt_versions %}
        <th>{{ ver }}</th>
        {% endfor %}
      </tr>
    </thead>
    <tbody>
      {% for model in report.models %}
      <tr>
        <th>{{ model }}</th>
        {% for ver in report.prompt_versions %}
        {% set cell = cell_map.get((model, ver)) %}
        <td>
          {% if cell %}
          <div class="cell-rate">
            {% if cell.pass_rate == 100.0 %}
            <span class="tag tag-ok">{{ "%.1f" | format(cell.pass_rate) }}%</span>
            {% elif cell.pass_rate >= 50.0 %}
            <span class="tag tag-warn">{{ "%.1f" | format(cell.pass_rate) }}%</span>
            {% else %}
            <span class="tag tag-err">{{ "%.1f" | format(cell.pass_rate) }}%</span>
            {% endif %}
            ({{ cell.passed }}/{{ cell.total }})
          </div>
          <div class="cell-sub">平均延迟：{{ "%.2f" | format(cell.avg_latency_ms) }} ms</div>
          <div class="cell-sub">总 Token：{{ cell.total_tokens }}</div>
          <div class="cell-sub">网络错/重试：{{ cell.network_errors }}/{{ cell.network_retries }} ｜ 校验错/自纠：{{ cell.validation_errors }}/{{ cell.validation_retries }}</div>
          {% else %}
          <span>-</span>
          {% endif %}
        </td>
        {% endfor %}
      </tr>
      {% endfor %}
    </tbody>
  </table>

  <h2>失败用例明细（共 {{ failed_cases | length }} 条）</h2>
  {% if failed_cases %}
  <table>
    <thead>
      <tr>
        <th>模型</th>
        <th>提示词版本</th>
        <th>用例 ID</th>
        <th>失败类型 / 步骤</th>
        <th>失败原因</th>
        <th>原始输出</th>
      </tr>
    </thead>
    <tbody>
      {% for item in failed_cases %}
      <tr>
        <td>{{ item.model }}</td>
        <td>{{ item.prompt_version }}</td>
        <td>{{ item.case_id }}</td>
        <td>
          <span class="tag tag-err">{{ "网络/服务错误" if item.failure_type == "network_error" else "输出校验失败" }}</span>
          <div class="cell-sub">步骤：{{ item.failed_step or "-" }}</div>
          <div class="cell-sub">重试：网络 {{ item.network_retries }} / 自纠 {{ item.validation_retries }}</div>
        </td>
        <td>{{ item.failure_reason or "-" }}</td>
        <td><pre class="raw-output">{{ item.raw_output if item.raw_output is not none else "(无输出)" }}</pre></td>
      </tr>
      {% endfor %}
    </tbody>
  </table>
  {% else %}
  <div class="empty-box">全部用例均已通过，无失败记录。</div>
  {% endif %}
</body>
</html>
"""


def load_run_report(json_path: Path) -> RunReportData:
    """从 runs/<时间戳>.json 读取并反序列化运行结果。"""
    return RunReportData.model_validate_json(json_path.read_text(encoding="utf-8"))


def format_matrix_text(report: RunReportData) -> str:
    """生成便于终端/文本阅读的「模型 × 提示词版本」矩阵文本。"""
    cell_map: dict[tuple[str, str], MatrixCellSummary] = {
        (c.model, c.prompt_version): c for c in report.matrix
    }
    lines = ["模型 | " + " | ".join(report.prompt_versions)]
    lines.append("-" * 60)
    for model in report.models:
        row_parts = [model]
        for ver in report.prompt_versions:
            cell = cell_map.get((model, ver))
            if cell is None:
                row_parts.append("-")
            else:
                row_parts.append(
                    f"{cell.pass_rate:.1f}% ({cell.passed}/{cell.total}), "
                    f"avg {cell.avg_latency_ms:.2f}ms, tok {cell.total_tokens}"
                )
        lines.append(" | ".join(row_parts))
    return "\n".join(lines)


def render_html_report(report: RunReportData) -> str:
    """使用 Jinja2 渲染完整静态 HTML 报告字符串。"""
    env = Environment(autoescape=select_autoescape(["html", "xml"]))
    template = env.from_string(_HTML_TEMPLATE)
    cell_map = {(c.model, c.prompt_version): c for c in report.matrix}
    failed_cases = [r for r in report.results if not r.passed]
    return template.render(
        report=report,
        cell_map=cell_map,
        failed_cases=failed_cases,
    )


def generate_html_report(
    run_json_path: Path,
    output_path: Path | None = None,
) -> tuple[RunReportData, Path]:
    """读取运行结果 JSON 并写出同级静态 HTML 报告。"""
    report = load_run_report(run_json_path)
    html_content = render_html_report(report)
    target = output_path or run_json_path.with_suffix(".html")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(html_content, encoding="utf-8")
    return report, target


def _find_latest_run_json(runs_dir: Path = RUNS_DIR) -> Path:
    """查找 runs 目录下最新的运行结果 JSON 文件。"""
    files = sorted(runs_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"目录 {runs_dir} 下没有找到运行结果 JSON 文件")
    return files[0]


def main(argv: list[str] | None = None) -> int:
    """命令行入口：将 runs/*.json 渲染为静态 HTML 报告。"""
    parser = argparse.ArgumentParser(description="PromptLab HTML 报告生成器")
    parser.add_argument("run_json", nargs="?", type=Path, help="运行结果 JSON 文件路径")
    parser.add_argument("--latest", action="store_true", help="使用 runs/ 下最新的 JSON 文件")
    parser.add_argument("--output", "-o", type=Path, help="指定输出 HTML 路径")
    args = parser.parse_args(argv)

    if args.latest or args.run_json is None:
        json_path = _find_latest_run_json()
    else:
        json_path = args.run_json

    report, html_path = generate_html_report(json_path, output_path=args.output)
    print(f"HTML 报告已生成: {html_path}")
    print("\n=== 矩阵摘要 ===")
    print(format_matrix_text(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
