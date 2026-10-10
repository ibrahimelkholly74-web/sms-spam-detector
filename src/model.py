import tensorflow as tf

VOCAB_SIZE = 10000
MAX_LEN = 50


def build_model(train_texts) -> tf.keras.Model:
    vectorize = tf.keras.layers.TextVectorization(
        max_tokens=VOCAB_SIZE, output_sequence_length=MAX_LEN
    )
    vectorize.adapt(tf.data.Dataset.from_tensor_slices(train_texts).batch(256))

    model = tf.keras.Sequential(
        [
            vectorize,
            tf.keras.layers.Embedding(VOCAB_SIZE, 64, mask_zero=True),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
    )
    return model