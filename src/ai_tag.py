import os
import json
import time
import pandas as pd
from openai import OpenAI

# Keep labels aligned with your existing tags
LABELS = [
    "cloud_partnerships",
    "ai_demand", 
    "infrastructure_expansion",
    "product_launch",
    "earnings",
    "competition",
    "regulation_export",
    "other",
]

SYSTEM_INSTRUCTIONS = f"""
You classify GPU/AI-related news headlines into exactly ONE label.

Allowed labels: {", ".join(LABELS)}

Definitions:
- cloud_partnerships: AWS/Azure/GCP, partnerships, platform integrations
- ai_demand: demand, backlog, orders, capacity constraints, adoption
- infrastructure_expansion: datacenter buildouts, capex, factories, supply chain
- product_launch: launches/releases/announcements of GPUs, chips, platforms
- earnings: earnings, guidance, revenue, margins, quarterly results
- competition: AMD/Intel/other competitor positioning, comparisons
- regulation_export: export controls, sanctions, regulation, China policy
- other: none of the above

Return ONLY valid JSON:
{{"label":"<one label>", "confidence": <0..1>, "reason":"<=12 words">}}
"""

def safe_json_parse(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        # Fallback: try to extract JSON object if model added extra text
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except Exception:
                pass
    return {"label": "other", "confidence": 0.0, "reason": "parse_failed"}

def classify_title(client: OpenAI, title: str, model: str = "gpt-4o-mini") -> dict:
    # Responses API (recommended for new projects)
    resp = client.responses.create(
        model=model,
        instructions=SYSTEM_INSTRUCTIONS,
        input=f"Headline: {title}",
    )
    obj = safe_json_parse(resp.output_text)

    label = str(obj.get("label", "other")).strip()
    conf = float(obj.get("confidence", 0.0) or 0.0)
    reason = str(obj.get("reason", ""))[:80]

    if label not in LABELS:
        label, conf, reason = "other", 0.0, "invalid_label"

    # clamp
    conf = max(0.0, min(1.0, conf))
    return {"ai_label": label, "ai_confidence": conf, "ai_reason": reason}

def main(
    in_path: str = "data/signals.csv",
    out_path: str = "data/signals_ai.csv",
    min_conf: float = 0.60,
    sleep_s: float = 0.1,
):
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set. Add it as a Codespaces secret and restart the Codespace.")

    df = pd.read_csv(in_path)
    client = OpenAI()

    # Only AI-classify the ambiguous bucket (other)
    mask_other = df["tag"].astype(str).eq("other")

    df["ai_label"] = ""
    df["ai_confidence"] = 0.0
    df["ai_reason"] = ""
    df["final_tag"] = df["tag"]

    other_idx = df.index[mask_other].tolist()
    print(f"Found {len(other_idx)} rows tagged 'other' to AI-classify...")

    for i, idx in enumerate(other_idx, start=1):
        title = str(df.at[idx, "title"]).strip()
        if not title:
            continue

        try:
            result = classify_title(client, title)
        except Exception as e:
            # Basic resilience: skip and keep 'other'
            df.at[idx, "ai_label"] = "other"
            df.at[idx, "ai_confidence"] = 0.0
            df.at[idx, "ai_reason"] = f"error:{type(e).__name__}"
            continue

        df.at[idx, "ai_label"] = result["ai_label"]
        df.at[idx, "ai_confidence"] = result["ai_confidence"]
        df.at[idx, "ai_reason"] = result["ai_reason"]

        # Promote AI label only if confidence is high enough
        if result["ai_confidence"] >= min_conf and result["ai_label"] != "other":
            df.at[idx, "final_tag"] = result["ai_label"]

        if i % 10 == 0 or i == len(other_idx):
            print(f"Processed {i}/{len(other_idx)} 'other' rows...")

        time.sleep(sleep_s)

    df.to_csv(out_path, index=False)

    print(f"\nSaved AI-tagged dataset to: {out_path}")
    print("\nFinal tag distribution:")
    print(df["final_tag"].value_counts())

if __name__ == "__main__":
    main()
