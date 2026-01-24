import os
import json
import pandas as pd
from datetime import timedelta
from openai import OpenAI

def load_recent(df: pd.DataFrame, days: int) -> pd.DataFrame:
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")
    max_date = df["published_at"].max()
    if pd.isna(max_date):
        return df.iloc[0:0]
    return df[df["published_at"] >= (max_date - timedelta(days=days))].copy()

def format_headlines(df: pd.DataFrame, limit_per_tag: int = 2) -> str:
    # Keep it compact for the model: top tags + a few headlines each
    out = []
    tag_col = "final_tag" if "final_tag" in df.columns else "tag"

    # Weight by simple counts here; you can swap in weights later
    tag_counts = df[tag_col].value_counts().head(4)

    for tag in tag_counts.index:
        out.append(f"\n## {tag} (count={int(tag_counts[tag])})")
        subset = df[df[tag_col] == tag].sort_values("published_at", ascending=False).head(limit_per_tag)
        for _, r in subset.iterrows():
            title = str(r.get("title", "")).strip()
            source = str(r.get("source", "")).strip()
            url = str(r.get("url", "")).strip()
            out.append(f"- [{source}] {title} ({url})")
    return "\n".join(out)

def summarize_with_ai(client: OpenAI, briefing_input: str, mode: str) -> dict:
    # mode: "daily" or "weekly"
    instructions = f"""
You are writing a {mode} GPU/NVIDIA market briefing based ONLY on the headlines provided.
Do NOT invent facts. Do NOT add numbers or claims not supported by the titles.

Return ONLY valid JSON with keys:
- executive_summary: 3 bullet points
- what_changed: 3 bullet points
- watchlist: 3 bullet points
- notable_headlines: 5 bullet points (each must reference a provided headline)
Keep each bullet <= 18 words.
"""

    resp = client.responses.create(
        model="gpt-4o-mini",
        instructions=instructions,
        input=briefing_input
    )
    text = resp.output_text.strip()
    try:
        return json.loads(text)
    except Exception:
        # Fallback: wrap raw text
        return {
            "executive_summary": ["(parse_failed)"],
            "what_changed": [],
            "watchlist": [],
            "notable_headlines": [text[:200]]
        }

def write_markdown(out_path: str, mode: str, days: int, tag_counts: pd.Series, summary_json: dict, raw_headlines: str):
    lines = []
    lines.append(f"# GPU Signal Briefing ({mode})")
    lines.append(f"_Window: last {days} day(s)_\n")

    lines.append("## Tag snapshot")
    for tag, c in tag_counts.items():
        lines.append(f"- **{tag}**: {int(c)}")

    lines.append("\n## Executive summary")
    for b in summary_json.get("executive_summary", []):
        lines.append(f"- {b}")

    lines.append("\n## What changed")
    for b in summary_json.get("what_changed", []):
        lines.append(f"- {b}")

    lines.append("\n## Watchlist")
    for b in summary_json.get("watchlist", []):
        lines.append(f"- {b}")

    lines.append("\n## Notable headlines")
    for b in summary_json.get("notable_headlines", []):
        lines.append(f"- {b}")

    # Optional: include the raw structured input for transparency
    lines.append("\n---\n## Source headlines (input)")
    lines.append(raw_headlines)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main(mode: str = "weekly"):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set (Codespaces secret).")

    # Choose your AI-tagged dataset if present
    in_path = "data/signals_ai.csv" if os.path.exists("data/signals_ai.csv") else "data/signals.csv"
    df = pd.read_csv(in_path)

    days = 1 if mode == "daily" else 7
    recent = load_recent(df, days=days)

    if recent.empty:
        print("No recent rows found. Run fetch_rss.py first.")
        return

    tag_col = "final_tag" if "final_tag" in recent.columns else "tag"
    tag_counts = recent[tag_col].value_counts().head(10)

    raw_headlines = format_headlines(recent, limit_per_tag=4)
    briefing_input = f"Headlines grouped by tag:\n{raw_headlines}"

    client = OpenAI()
    summary_json = summarize_with_ai(client, briefing_input, mode=mode)

    out_path = f"data/briefing_{mode}.md"
    write_markdown(out_path, mode, days, tag_counts, summary_json, raw_headlines)
    print(f"Saved briefing to {out_path}")

if __name__ == "__main__":
    # Run: python src/summarize_brief.py daily
    import sys
    mode = sys.argv[1].strip().lower() if len(sys.argv) > 1 else "weekly"
    if mode not in ("daily", "weekly"):
        raise SystemExit("Usage: python src/summarize_brief.py [daily|weekly]")
    main(mode)
