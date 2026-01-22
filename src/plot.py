import pandas as pd
import matplotlib.pyplot as plt

def main(path="data/signals.csv", out="data/tag_counts.png"):
    df = pd.read_csv(path)
    df["published_at"] = pd.to_datetime(df["published_at"], utc=True, errors="coerce")

    recent = df[df["published_at"] >= (df["published_at"].max() - pd.Timedelta(days=7))]
    counts = recent["tag"].value_counts()

    plt.figure()
    counts.plot(kind="bar")
    plt.title("GPU Signal Tracker — Tag counts (last 7 days)")
    plt.xlabel("Tag")
    plt.ylabel("Articles")
    plt.tight_layout()
    plt.savefig(out, dpi=200)

if __name__ == "__main__":
    main()
