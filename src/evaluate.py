import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.config import ARTIFACTS_DIR
from src.data import load_dataset, split_dataset


def predict_probs(model, texts) -> np.ndarray:
    return model.predict(tf.constant(list(texts)), verbose=0).ravel()


def find_best_threshold(y_true, probs) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[np.argmax(f1)])


def plot_training_curves(history: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history["loss"], label="train")
    axes[0].plot(history["val_loss"], label="val")
    axes[0].set_title("Loss")
    axes[0].legend()
    axes[1].plot(history["accuracy"], label="train")
    axes[1].plot(history["val_accuracy"], label="val")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(ARTIFACTS_DIR / "training_curves.png", dpi=150)
    plt.close(fig)


def main():
    model = tf.keras.models.load_model(ARTIFACTS_DIR / "model.keras")
    splits = split_dataset(load_dataset())

    val_probs = predict_probs(model, splits.X_val)
    threshold = find_best_threshold(splits.y_val, val_probs)
    print(f"Threshold chosen on validation: {threshold:.4f}")

    test_probs = predict_probs(model, splits.X_test)
    y_pred = (test_probs > threshold).astype(int)

    print(classification_report(splits.y_test, y_pred, target_names=["ham", "spam"]))

    ConfusionMatrixDisplay.from_predictions(
        splits.y_test, y_pred, display_labels=["ham", "spam"], cmap="Blues"
    )
    plt.title(f"Test confusion matrix (threshold={threshold:.2f})")
    plt.savefig(ARTIFACTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()

    history_path = ARTIFACTS_DIR / "history.json"
    if history_path.exists():
        plot_training_curves(json.loads(history_path.read_text()))

    metrics = {
        "threshold": threshold,
        "spam_precision": float(precision_score(splits.y_test, y_pred)),
        "spam_recall": float(recall_score(splits.y_test, y_pred)),
        "spam_f1": float(f1_score(splits.y_test, y_pred)),
        "macro_f1": float(f1_score(splits.y_test, y_pred, average="macro")),
    }
    (ARTIFACTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2))

    errors = pd.DataFrame(
        {
            "text": splits.X_test,
            "true": splits.y_test,
            "pred": y_pred,
            "spam_prob": test_probs,
        }
    )
    errors = errors[errors["true"] != errors["pred"]].sort_values(
        "spam_prob", ascending=False
    )
    errors.to_csv(ARTIFACTS_DIR / "errors.csv", index=False, encoding="utf-8")

    print("Spam F1: ", round(metrics["spam_f1"], 4))
    print("Macro F1:", round(metrics["macro_f1"], 4))
    print("Errors saved:", len(errors), "->", ARTIFACTS_DIR / "errors.csv")


if __name__ == "__main__":
    main()