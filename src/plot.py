import pandas as pd
import matplotlib.pyplot as plt
from datetime import timedelta

# 1️⃣ Source importance (business logic)
SOURCE_WEIGHTS = {
    "NVIDIA Blog": 3,
    "The Verge": 1,
}

def main(path="data/signals_ai.csv", out="data/tag_counts.png"):
    df = pd.read_csv(path)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    # 2️⃣ Time filter — last 7 days only
    max_date = df["published_at"].max()
    recent_df = df[df["published_at"] >= (max_date - timedelta(days=7))]

    # 3️⃣ Apply source weights
    recent_df["weight"] = recent_df["source"].map(SOURCE_WEIGHTS).fillna(1)

    # 4️⃣ Weighted count by tag
    weighted_counts = (
        recent_df
        .groupby("final_tag")["weight"]
        .sum()
        .sort_values(ascending=False)
    )

    # 5️⃣ Plot
    plt.figure()
    weighted_counts.plot(kind="bar")
    plt.title("Weighted GPU Signal Momentum (Last 7 Days)")
    plt.xlabel("Tag")
    plt.ylabel("Weighted Mentions")
    plt.tight_layout()
    plt.savefig(out, dpi=200)

if __name__ == "__main__":
    main()

