"""Parse LawRefBook-style Markdown law files into structured documents."""

import json
import re
from pathlib import Path
from typing import TypedDict


class ParsedArticle(TypedDict):
    article_no: str
    content: str
    chapter: str | None


class ParsedLaw(TypedDict):
    law_id: str
    title: str
    publish_date: str | None
    full_text: str
    articles: list[ParsedArticle]


# 必须在行首才算条款(避免正文里"第X条"的误匹配)
ARTICLE_PATTERN = re.compile(r"(?:^|\n)\s*(第[一二三四五六七八九十百千零0-9]+条)")

CHAPTER_PATTERN = re.compile(r"(?:^|\n)#{0,6}\s*(第[一二三四五六七八九十百千]+章\s*[^\n]+)")

SKIP_FILENAME_PATTERNS = [
    "_index",
    "README",
    "readme",
    "目录",
    "索引",
    "贡献",
    "指南",
]

# 路径必须包含其中一个目录才处理(白名单)
INCLUDE_PATH_KEYWORDS = [
    "民法典",
    "刑法",
    "宪法",
    "宪法相关法",
    "民法商法",
    "社会法",
    "经济法",
    "行政法",
    "诉讼与非诉讼",
]

# 路径含其中任一关键词就跳过(黑名单,优先级高于白名单)
EXCLUDE_PATH_KEYWORDS = [
    "DLC",
    "地方法规",
    "司法解释",
    "案例",
    "部门规章",
    "行政法规",
]

MIN_CONTENT_LENGTH = 300
MIN_ARTICLES = 3


def strip_frontmatter(text: str) -> str:
    if text.startswith("---"):
        match = re.match(r"^---\n.*?\n---\n", text, re.DOTALL)
        if match:
            return text[match.end() :]
    return text


def extract_title(text: str, fallback: str) -> str:
    match = re.search(r"^#\s+(.+?)$", text, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return fallback


def split_into_articles(text: str) -> list[ParsedArticle]:
    articles: list[ParsedArticle] = []
    matches = list(ARTICLE_PATTERN.finditer(text))
    if not matches:
        return articles

    for i, match in enumerate(matches):
        article_no = match.group(1)
        start = match.start(1)
        end = matches[i + 1].start(1) if i + 1 < len(matches) else len(text)

        current_chapter: str | None = None
        chapter_search = list(CHAPTER_PATTERN.finditer(text[:start]))
        if chapter_search:
            current_chapter = chapter_search[-1].group(1).strip()

        article_text = text[start:end].strip()
        if len(article_text) < 5:
            continue

        articles.append(
            ParsedArticle(
                article_no=article_no,
                content=article_text,
                chapter=current_chapter,
            )
        )
    return articles


def should_skip_file(path: Path) -> bool:
    """Skip rules:
    1. Skip _index, README, etc.
    2. Skip if path contains any EXCLUDE keyword
    3. Skip if path does NOT contain any INCLUDE keyword
    """
    name = path.stem.lower()
    for pattern in SKIP_FILENAME_PATTERNS:
        if pattern.lower() in name:
            return True

    path_str = str(path)

    # 黑名单优先
    for excl in EXCLUDE_PATH_KEYWORDS:
        if excl in path_str:
            return True

    # 白名单
    if not any(inc in path_str for inc in INCLUDE_PATH_KEYWORDS):
        return True

    return False


def parse_markdown_file(path: Path) -> ParsedLaw | None:
    if should_skip_file(path):
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"  Cannot read {path.name}: {e}")
        return None
    text = strip_frontmatter(text)
    if len(text.strip()) < MIN_CONTENT_LENGTH:
        return None
    title = extract_title(text, path.stem)
    articles = split_into_articles(text)
    if len(articles) < MIN_ARTICLES:
        return None
    return ParsedLaw(
        law_id=path.stem,
        title=title,
        publish_date=None,
        full_text=text,
        articles=articles,
    )


def parse_all(input_dir: Path, output_dir: Path) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    skipped = 0
    md_files = list(input_dir.rglob("*.md"))
    print(f"Found {len(md_files)} .md files. Parsing (core categories only)...\n")
    for raw_path in md_files:
        parsed = parse_markdown_file(raw_path)
        if parsed is None:
            skipped += 1
            continue
        safe_name = "".join(c for c in parsed["title"] if c.isalnum() or c in "._-()")[:80]
        if not safe_name:
            safe_name = parsed["law_id"]
        out_path = output_dir / f"{safe_name}.parsed.json"
        if out_path.exists():
            safe_name = f"{safe_name}_{count}"
            out_path = output_dir / f"{safe_name}.parsed.json"
        out_path.write_text(
            json.dumps(parsed, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if count < 50:
            print(f"  {parsed['title'][:40]}: {len(parsed['articles'])} articles")
        count += 1
    print(f"\nParsed {count} laws into {output_dir}")
    print(f"Skipped {skipped} files")
    return count


if __name__ == "__main__":
    n = parse_all(Path("data/raw"), Path("data/parsed"))
