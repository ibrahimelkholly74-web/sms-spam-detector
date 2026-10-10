import json

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

from src.config import ARTIFACTS_DIR, SEED
from src.data import load_dataset, split_dataset
from src.evaluate import find_best_threshold


def make_tfidf() -> TfidfVectorizer:
    return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def build_baselines() -> dict:
    return {
        "TF-IDF + LogReg": make_pipeline(
            make_tfidf(),
            LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=SEED
            ),
        ),
        "TF-IDF + NaiveBayes": make_pipeline(make_tfidf(), MultinomialNB(alpha=0.1)),
    }


def score(y_true, y_pred) -> dict:
    return {
        "spam_precision": precision_score(y_true, y_pred),
        "spam_recall": recall_score(y_true, y_pred),
        "spam_f1": f1_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "errors": int((y_true != y_pred).sum()),
    }


def main():
    splits = split_dataset(load_dataset())
    rows = {}

    for name, pipe in build_baselines().items():
        pipe.fit(splits.X_train, splits.y_train)

        val_probs = pipe.predict_proba(splits.X_val)[:, 1]
        threshold = find_best_threshold(splits.y_val, val_probs)

        test_probs = pipe.predict_proba(splits.X_test)[:, 1]
        y_pred = (test_probs > threshold).astype(int)
        rows[name] = score(splits.y_test, y_pred)

    metrics_path = ARTIFACTS_DIR / "metrics.json"
    errors_path = ARTIFACTS_DIR / "errors.csv"
    if metrics_path.exists() and errors_path.exists():
        m = json.loads(metrics_path.read_text())
        rows["BiLSTM"] = {
            "spam_precision": m["spam_precision"],
            "spam_recall": m["spam_recall"],
            "spam_f1": m["spam_f1"],
            "macro_f1": m["macro_f1"],
            "errors": len(pd.read_csv(errors_path)),
        }

    table = pd.DataFrame(rows).T.round(4)
    table["errors"] = table["errors"].astype(int)
    table.index.name = "model"
    table.to_csv(ARTIFACTS_DIR / "comparison.csv")
    print(table.to_string())


if __name__ == "__main__":
    main()