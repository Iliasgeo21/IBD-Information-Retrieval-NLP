from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from preprocessing import load_pubmed200k_rct, filter_by_keywords
from utils import ensure_dir, save_json, save_text, plot_confusion, classification_report_str

def build_model(use_url: str, num_classes: int, lr: float = 1e-3) -> tf.keras.Model:
    encoder = hub.KerasLayer(use_url, input_shape=[], dtype=tf.string, trainable=False, name="use_encoder")
    inputs = tf.keras.Input(shape=(), dtype=tf.string, name="text")
    x = encoder(inputs)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def run(data_dir: str, out_dir: str, use_url: str, epochs: int = 5, batch_size: int = 64, seed: int = 42):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "confusion_matrices")

    # 1) Load & filter
    *_ , _, all_df = load_pubmed200k_rct(data_dir)
    filtered = filter_by_keywords(all_df)

    texts = filtered["text"].astype(str).tolist()
    labels = filtered["target"].astype(str).tolist()

    enc = LabelEncoder()
    y = enc.fit_transform(labels)
    class_names = enc.classes_.tolist()

    X_train, X_tmp, y_train, y_tmp = train_test_split(texts, y, test_size=0.2, random_state=seed, stratify=y)
    X_val, X_test,  y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=seed, stratify=y_tmp)

    # 2) Model
    model = build_model(use_url, num_classes=len(class_names))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=2, restore_best_weights=True)
    ]

    # 3) Train
    hist = model.fit(
        x=np.array(X_train), y=np.array(y_train),
        validation_data=(np.array(X_val), np.array(y_val)),
        epochs=epochs, batch_size=batch_size, callbacks=callbacks, verbose=1
    )

    # 4) Evaluate
    test_probs = model.predict(np.array(X_test), batch_size=batch_size, verbose=0)
    y_pred = np.argmax(test_probs, axis=1)

    acc = (y_pred == y_test).mean().item()
    report = classification_report_str(y_test, y_pred, labels=list(range(len(class_names))))
    save_text(report, out_dir / "use_test_report.txt")
    plot_confusion(y_test, y_pred, labels=list(range(len(class_names))),
                   out_path=out_dir / "confusion_matrices" / "use_confusion_matrix.png")

    save_json({"test_accuracy": acc, "labels": class_names}, out_dir / "use_metrics.json")
    model.save(out_dir / "use_model_keras")

    print("Done. USE metrics written to:", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="results")
    ap.add_argument("--use_url", type=str, default="https://tfhub.dev/google/universal-sentence-encoder/4")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.data_dir, args.out_dir, args.use_url, args.epochs, args.batch_size, args.seed)
