import sys

import pandas as pd

from src.config import ARTIFACTS_DIR

MAX_CHARS = 150


def show(title: str, df: pd.DataFrame) -> None:
    print(f"\n=== {title} ({len(df)}) ===")
    for row in df.itertuples():
        text = row.text.replace("\n", " ")
        if len(text) > MAX_CHARS:
            text = text[:MAX_CHARS] + "..."
        print(f"[{row.spam_prob:.2f}] {text}")


def main():
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    errors = pd.read_csv(ARTIFACTS_DIR / "errors.csv")

    false_positives = errors[(errors["true"] == 0) & (errors["pred"] == 1)]
    false_negatives = errors[(errors["true"] == 1) & (errors["pred"] == 0)]

    show("False Positives: real ham, predicted spam", false_positives)
    show("False Negatives: real spam, missed", false_negatives)


if __name__ == "__main__":
    main()