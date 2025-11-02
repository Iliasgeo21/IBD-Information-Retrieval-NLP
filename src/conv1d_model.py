
-------------
This script trains a Conv1D neural network to classify PubMed abstract sentences
into rhetorical categories (BACKGROUND, OBJECTIVE, METHODS, RESULTS, CONCLUSIONS).

It uses:
- TensorFlow TextVectorization
- Embedding + Conv1D + GlobalMaxPool
- Train / Validation / Test split
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import os
import argparse
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 🔧 Load Data
# ============================================

def load_data(data_dir):
    train = pd.read_csv(os.path.join(data_dir, "train.txt"), sep="\t", header=None, names=["label", "text"])
    val   = pd.read_csv(os.path.join(data_dir, "dev.txt"), sep="\t", header=None, names=["label", "text"])
    test  = pd.read_csv(os.path.join(data_dir, "test.txt"), sep="\t", header=None, names=["label", "text"])
    return train, val, test

# ============================================
# 🧠 Build Conv1D Model
# ============================================

def create_model(vocab_size=20000, embedding_dim=128, seq_length=290):

    text_vectorization = tf.keras.layers.TextVectorization(
        max_tokens=vocab_size,
        output_sequence_length=seq_length
    )

    model = tf.keras.Sequential([
        text_vectorization,
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Conv1D(64, 5, activation="relu"),
        tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(5, activation="softmax")
    ])

    model.compile(
        loss="sparse_categorical_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    return model, text_vectorization

# ============================================
# 📊 Training & Evaluation
# ============================================

def train_and_evaluate(data_dir, out_dir):

    os.makedirs(out_dir, exist_ok=True)

    train, val, test = load_data(data_dir)

    label_map = {label: idx for idx, label in enumerate(train["label"].unique())}

    y_train = train["label"].map(label_map).values
    y_val   = val["label"].map(label_map).values
    y_test  = test["label"].map(label_map).values

    model, vectorizer = create_model()

    vectorizer.adapt(train["text"].values)

    history = model.fit(
        train["text"].values, y_train,
        validation_data=(val["text"].values, y_val),
        batch_size=128,
        epochs=10
    )

    # Save model
    model.save(os.path.join(out_dir, "conv1d_model.h5"))

    # Predictions
    preds = np.argmax(model.predict(test["text"].values), axis=1)

    # Classification Report
    report = classification_report(y_test, preds, target_names=label_map.keys())
    print(report)

    with open(os.path.join(out_dir, "classification_report_conv1d.txt"), "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Conv1D Test Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(out_dir, "confusion_matrix_conv1d.png"))
    plt.close()

    # Training Curves
    plt.figure(figsize=(8,5))
    plt.plot(history.history["loss"], label="Train Loss")
    plt.plot(history.history["val_loss"], label="Val Loss")
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Conv1D Learning Curve")
    plt.xlabel("Epochs")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.savefig(os.path.join(out_dir, "learning_curve_conv1d.png"))
    plt.close()


# ============================================
# 🚀 CLI
# ============================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    train_and_evaluate(args.data_dir, args.out_dir)

