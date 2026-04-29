"""Download law documents from the National Laws and Regulations Database."""

import json
import time
from pathlib import Path

import httpx

# 国家法规库的 API 端点
SEARCH_API = "https://flk.npc.gov.cn/api/"

# 下载这些类别的法律
KEYWORDS = [
    "民法典",
    "合同",
    "劳动",
    "公司法",
    "消费者权益",
    "婚姻",
    "继承",
    "侵权责任",
    "物权",
    "知识产权",
]

OUTPUT_DIR = Path("data/raw")


def search_laws(keyword: str, page: int = 1, size: int = 10) -> list[dict]:
    """Search laws by keyword."""
    params = {
        "type": "flfgmc",
        "searchType": "title;accurate",
        "sortTr": "f_bbrq_s;desc",
        "gbrqStart": "",
        "gbrqEnd": "",
        "sxrqStart": "",
        "sxrqEnd": "",
        "sort": "true",
        "page": page,
        "size": size,
        "_": int(time.time() * 1000),
        "title": keyword,
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (legal-agent-learning-project)",
    }
    try:
        resp = httpx.get(SEARCH_API, params=params, headers=headers, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("result", {}).get("data", [])
    except Exception as e:
        print(f"  ❌ Search failed for '{keyword}': {e}")
        return []


def download_law_detail(law_id: str, title: str) -> dict | None:
    """Download a single law's full text."""
    detail_api = "https://flk.npc.gov.cn/api/detail"
    params = {"id": law_id}
    headers = {"User-Agent": "Mozilla/5.0 (legal-agent-learning-project)"}

    try:
        resp = httpx.get(detail_api, params=params, headers=headers, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  ❌ Download failed for '{title}': {e}")
        return None


def main() -> None:
    """Run the download pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    total_downloaded = 0

    for keyword in KEYWORDS:
        print(f"\n🔍 Searching: {keyword}")
        results = search_laws(keyword, size=5)
        print(f"   Found {len(results)} laws")

        for law in results:
            law_id = law.get("id")
            title = law.get("title", "unknown")
            if not law_id:
                continue

            # 文件名清理
            safe_title = "".join(c for c in title if c.isalnum() or c in "._-")[:80]
            output_path = OUTPUT_DIR / f"{safe_title}.json"

            if output_path.exists():
                print(f"   ⏭️  Already exists: {title}")
                continue

            print(f"   ⬇️  Downloading: {title}")
            detail = download_law_detail(law_id, title)
            if detail:
                output_path.write_text(
                    json.dumps(detail, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                total_downloaded += 1
                time.sleep(0.5)  # 礼貌爬虫,别打挂人家服务器

    print(f"\n✅ Done. Downloaded {total_downloaded} laws to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
