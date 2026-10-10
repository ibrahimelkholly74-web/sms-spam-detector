import json

import tensorflow as tf

from src.config import ARTIFACTS_DIR, SEED
from src.data import load_dataset, split_dataset
from src.model import build_model

BATCH_SIZE = 32
EPOCHS = 15


def make_dataset(texts, labels, shuffle):
    ds = tf.data.Dataset.from_tensor_slices((texts, labels.astype("float32")))
    if shuffle:
        ds = ds.shuffle(len(texts), seed=SEED)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def main():
    tf.keras.utils.set_random_seed(SEED)

    splits = split_dataset(load_dataset())
    model = build_model(splits.X_train)

    n_ham = int((splits.y_train == 0).sum())
    n_spam = int((splits.y_train == 1).sum())
    class_weight = {0: 1.0, 1: n_ham / n_spam}

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )

    history = model.fit(
        make_dataset(splits.X_train, splits.y_train, shuffle=True),
        validation_data=make_dataset(splits.X_val, splits.y_val, shuffle=False),
        epochs=EPOCHS,
        class_weight=class_weight,
        callbacks=[early_stop],
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model.save(ARTIFACTS_DIR / "model.keras")

    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (ARTIFACTS_DIR / "history.json").write_text(json.dumps(history_dict, indent=2))

    print("Saved:", ARTIFACTS_DIR / "model.keras")
    print("Best val_loss:", round(min(history_dict["val_loss"]), 4))


if __name__ == "__main__":
    main()