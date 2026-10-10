import json
from functools import lru_cache

import tensorflow as tf

from src.config import ARTIFACTS_DIR


@lru_cache(maxsize=1)
def load_artifacts():
    model = tf.keras.models.load_model(ARTIFACTS_DIR / "model.keras")
    metrics = json.loads((ARTIFACTS_DIR / "metrics.json").read_text())
    return model, float(metrics["threshold"])


def predict(texts: list[str]) -> list[dict]:
    if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
        raise TypeError("texts must be a list of strings")
    if not texts:
        return []

    model, threshold = load_artifacts()
    probs = model.predict(tf.constant(texts), verbose=0).ravel()

    return [
        {
            "text": text,
            "label": "spam" if p > threshold else "ham",
            "spam_probability": round(float(p), 4),
        }
        for text, p in zip(texts, probs)
    ]


def main():
    print("SMS Spam Detector. Type a message, or 'q' to quit.\n")
    while True:
        try:
            msg = input("Message: ").strip()
        except EOFError:
            break
        if msg.lower() == "q":
            break
        if not msg:
            continue
        result = predict([msg])[0]
        print(f"-> {result['label'].upper()} | spam probability: {result['spam_probability']:.1%}\n")


if __name__ == "__main__":
    main()