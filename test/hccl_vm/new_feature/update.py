#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#!/usr/bin/env python3

"""自动生成 new_feature/README.md。

使用方式：

    # 扫描当前脚本所在目录并生成 README.md
    python3 update.py

    # 检查 README.md 是否与当前特性文档一致，不修改文件
    python3 update.py --check

    # 指定特性目录和输出文件
    python3 update.py --root /path/to/new_feature \
        --output /path/to/new_feature/README.md

扫描规则：

* 只扫描 root 的一级子目录。
* 只有包含 feature.md 的子目录才会被收录。
* 特性名称取 feature.md 的一级标题（# 标题）。
* 发布日期优先取 feature.md 中的“发布日期”，缺失时读取目录名中的日期。
* 摘要取 feature.md 的“摘要”章节内容。
* README 中的特性按发布日期倒序排列。

常用参数：

    --root PATH    指定要扫描的特性根目录。
    --output PATH  指定生成的 README 路径。
    --check        只检查 README 是否需要更新，内容不一致时返回 1。
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path


DATE_PATTERN = re.compile(r"(?<!\d)(20\d{2}-\d{2}-\d{2})(?!\d)")
RELEASE_DATE_PATTERN = re.compile(r"发布日期\s*[：:]\s*`?(20\d{2}-\d{2}-\d{2})`?")


@dataclass(frozen=True)
class Feature:
    directory: Path
    title: str
    release_date: str
    summary: str

    @property
    def sort_date(self) -> date:
        try:
            return date.fromisoformat(self.release_date)
        except ValueError:
            return date.min


def parse_title(lines: list[str], fallback: str) -> str:
    for line in lines:
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def parse_release_date(text: str, directory_name: str) -> str:
    match = RELEASE_DATE_PATTERN.search(text)
    if match:
        return match.group(1)

    match = DATE_PATTERN.search(directory_name)
    return match.group(1) if match else "未标注"


def parse_summary(lines: list[str]) -> str:
    summary_start = None
    for index, line in enumerate(lines):
        if line.strip() == "## 摘要":
            summary_start = index + 1
            break

    if summary_start is None:
        return "暂无摘要"

    summary_lines: list[str] = []
    for line in lines[summary_start:]:
        if line.startswith("## "):
            break
        if line.strip():
            summary_lines.append(line.strip())

    summary = " ".join(summary_lines)
    return summary or "暂无摘要"


def parse_feature(directory: Path) -> Feature:
    feature_path = directory / "feature.md"
    text = feature_path.read_text(encoding="utf-8")
    lines = text.splitlines()
    return Feature(
        directory=directory,
        title=parse_title(lines, directory.name),
        release_date=parse_release_date(text, directory.name),
        summary=parse_summary(lines),
    )


def collect_features(root: Path) -> list[Feature]:
    features: list[Feature] = []
    for directory in sorted(root.iterdir()):
        if not directory.is_dir() or not (directory / "feature.md").is_file():
            continue
        features.append(parse_feature(directory))

    return sorted(features, key=lambda item: (item.sort_date, item.title), reverse=True)


def escape_table_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ").strip()


def render_readme(root: Path, features: list[Feature]) -> str:
    lines = [
        "# 新特性列表",
        "",
        "> 本文件由 `update.py` 自动生成，请勿直接编辑。",
        "> 更新命令：`python3 update.py`",
        "",
        "## 特性总览",
        "",
        "| 发布日期 | 特性 | 摘要 |",
        "|----------|------|------|",
    ]

    if features:
        for feature in features:
            relative_path = (Path(feature.directory.name) / "feature.md").as_posix()
            title = escape_table_cell(feature.title)
            summary = escape_table_cell(feature.summary)
            lines.append(
                f"| {feature.release_date} | [{title}](./{relative_path}) | {summary} |"
            )
    else:
        lines.append("| - | 暂无特性文档 | - |")

    lines.extend(
        [
            "",
            "## 文档规范",
            "",
            "- 每个特性使用独立目录，并在目录中放置 `feature.md`。",
            "- `feature.md` 的一级标题作为特性名称，`发布日期` 和 `摘要` 用于生成总览。",
            "- 图片和其他说明材料放置在对应特性目录中。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate the new feature overview README")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="feature directory to scan (default: directory containing this script)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output README path (default: <root>/README.md)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="check whether README is up to date without modifying it",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    output = (args.output or root / "README.md").resolve()
    if not root.is_dir():
        print(f"error: feature root does not exist: {root}", file=sys.stderr)
        return 2

    try:
        content = render_readme(root, collect_features(root))
    except (OSError, UnicodeError) as exc:
        print(f"error: failed to read feature documents: {exc}", file=sys.stderr)
        return 2

    if args.check:
        current = output.read_text(encoding="utf-8") if output.exists() else ""
        if current != content:
            print(f"outdated: {output}", file=sys.stderr)
            return 1
        print(f"up to date: {output}")
        return 0

    output.write_text(content, encoding="utf-8")
    print(f"generated: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
