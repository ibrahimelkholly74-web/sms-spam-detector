import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src.config import ARTIFACTS_DIR

COLUMNS = {
    "spam_precision": "Spam precision",
    "spam_recall": "Spam recall",
    "spam_f1": "Spam F1",
    "macro_f1": "Macro F1",
    "errors": "Test errors",
}

HEADER_COLOR = "#1f3a5f"
BEST_ROW_COLOR = "#e6f4ea"
EDGE_COLOR = "#cccccc"


def load_comparison() -> pd.DataFrame:
    path = ARTIFACTS_DIR / "comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python -m src.baselines")

    df = pd.read_csv(path, index_col="model")
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"comparison.csv is missing columns: {sorted(missing)}")

    return df[list(COLUMNS)].sort_values("spam_f1", ascending=False)


def format_row(model: str, row: pd.Series) -> list[str]:
    values = [f"{row[col]:.3f}" for col in COLUMNS if col != "errors"]
    return [model, *values, str(int(row["errors"]))]


def main():
    df = load_comparison()
    cell_text = [format_row(model, row) for model, row in df.iterrows()]
    headers = ["Model", *COLUMNS.values()]

    fig, ax = plt.subplots(figsize=(11, 1.6 + 0.6 * len(df)))
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        colLabels=headers,
        colWidths=[0.28] + [0.144] * len(COLUMNS),
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1, 2.0)

    for (row_idx, _col_idx), cell in table.get_celld().items():
        cell.set_edgecolor(EDGE_COLOR)
        if row_idx == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(color="white", fontweight="bold")
        elif row_idx == 1:
            cell.set_facecolor(BEST_ROW_COLOR)

    ax.set_title(
        "SMS spam detection: test-set results",
        fontsize=16,
        fontweight="bold",
        pad=14,
    )
    fig.text(
        0.5,
        0.02,
        "Thresholds tuned on a validation set. Highlighted row = highest spam F1.",
        ha="center",
        fontsize=10,
        color="#555555",
    )

    out_path = ARTIFACTS_DIR / "comparison_table.png"
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()