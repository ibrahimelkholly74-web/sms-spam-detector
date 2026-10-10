import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import KAGGLE_DATASET, RAW_CSV, SEED, TEST_SIZE, VAL_SIZE

LABEL_MAP = {"ham": 0, "spam": 1}


@dataclass(frozen=True)
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def ensure_raw_csv() -> Path:
    if RAW_CSV.exists():
        return RAW_CSV

    import kagglehub

    download_dir = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    matches = sorted(download_dir.rglob("*.csv"))
    if not matches:
        raise FileNotFoundError(f"No CSV found in {download_dir}")

    RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(matches[0], RAW_CSV)
    return RAW_CSV


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(ensure_raw_csv(), encoding="latin-1")
    missing = {"v1", "v2"} - set(df.columns)
    if missing:
        raise ValueError(f"Unexpected CSV schema, missing columns: {missing}")

    df = df[["v1", "v2"]].rename(columns={"v1": "label", "v2": "text"})
    df = df.dropna()
    df["text"] = df["text"].str.replace(r"[^\x00-\x7F]", "", regex=True).str.strip()
    df = df[df["text"] != ""]
    df = df.drop_duplicates(subset="text").reset_index(drop=True)
    df["label"] = df["label"].map(LABEL_MAP)

    if df["label"].isna().any():
        raise ValueError("Found labels outside {'ham', 'spam'}")

    df["label"] = df["label"].astype(int)
    return df


def split_dataset(df: pd.DataFrame) -> Splits:
    X = df["text"].to_numpy()
    y = df["label"].to_numpy()

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr,
        y_tr,
        test_size=VAL_SIZE / (1 - TEST_SIZE),
        random_state=SEED,
        stratify=y_tr,
    )
    return Splits(X_train, y_train, X_val, y_val, X_test, y_test)


if __name__ == "__main__":
    data = load_dataset()
    print("shape:", data.shape)
    print(data["label"].value_counts().to_dict())

    s = split_dataset(data)
    print("train/val/test:", len(s.X_train), len(s.X_val), len(s.X_test))
    print(
        "spam ratio:",
        round(s.y_train.mean(), 3),
        round(s.y_val.mean(), 3),
        round(s.y_test.mean(), 3),
    )