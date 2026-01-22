import feedparser
import pandas as pd
from dateutil import parser as dtparser
from datetime import datetime, timezone

FEEDS = [
    ("NVIDIA Blog", "https://blogs.nvidia.com/feed/"),
    ("The Verge", "https://www.theverge.com/rss/index.xml"),
]

KEYWORDS = {
    "earnings": ["earnings", "quarter", "guidance", "revenue"],
    "product_launch": ["launch", "announces", "release", "blackwell", "h200", "h100", "rtx", "geforce"],
    "datacenter_ai": ["datacenter", "data center", "inference", "training", "ai", "accelerator"],
    "competition": ["amd", "intel", "mi300", "gaudi"],
    "regulation_export": ["export", "restriction", "sanction", "china", "regulator"],
}

def tag_title(title: str) -> str:
    t = title.lower()
    for tag, words in KEYWORDS.items():
        if any(w in t for w in words):
            return tag
    return "other"

def parse_date(entry) -> str:
    for k in ["published", "updated"]:
        if k in entry:
            try:
                dt = dtparser.parse(entry[k]).astimezone(timezone.utc)
                return dt.isoformat()
            except Exception:
                pass
    return datetime.now(timezone.utc).isoformat()

def main(out_path="data/signals.csv"):
    rows = []
    for source, url in FEEDS:
        feed = feedparser.parse(url)
        for e in feed.entries[:50]:
            title = getattr(e, "title", "").strip()
            link = getattr(e, "link", "").strip()
            if not title or not link:
                continue
            rows.append({
                "published_at": parse_date(e),
                "source": source,
                "title": title,
                "url": link,
                "tag": tag_title(title),
            })

    df = pd.DataFrame(rows).drop_duplicates(subset=["title", "url"])
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    df = df.sort_values("published_at", ascending=False)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")

if __name__ == "__main__":
    main()
