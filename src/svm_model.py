from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from preprocessing import load_pubmed200k_rct, filter_by_keywords, clean_texts
from utils import ensure_dir, plot_learning_curve, plot_confusion, classification_report_str, save_text, save_json

def run(data_dir: str, out_dir: str, n_jobs: int = -1, random_state: int = 42):
    out_dir = Path(out_dir)
    ensure_dir(out_dir)
    ensure_dir(out_dir / "confusion_matrices")
    ensure_dir(out_dir / "learning_curves")

    # 1) Load & filter
    train_df, dev_df, test_df, full_train_df, all_df = load_pubmed200k_rct(data_dir)
    filtered = filter_by_keywords(all_df)  # κρατάμε κείμενα σχετικά με IBD/anti-TNF

    # 2) Labels
    y = filtered["target"].astype(str).values
    enc = LabelEncoder()
    y_enc = enc.fit_transform(y)
    class_names = enc.classes_.tolist()

    # 3) Text preprocessing
    X = clean_texts(filtered["text"].astype(str).tolist())

    # 4) Split (80/20 then 75/25 -> 60/20/20 περίπου)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y_enc, test_size=0.2, random_state=random_state, stratify=y_enc
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, test_size=0.25, random_state=random_state, stratify=y_train_full
    )

    # 5) Pipeline + search
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", SVC(random_state=random_state, probability=True, class_weight="balanced"))
    ])
    param_grid = {
        "tfidf__max_features": [10000],
        "tfidf__ngram_range": [(1, 2)],
        "clf__C": [0.1, 1, 10],
        "clf__gamma": ["scale", 0.01]
    }
    scorers = {
        "acc": "accuracy",
        "prec": "precision_weighted",
        "rec": "recall_weighted"
    }
    gs = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=n_jobs,
                      scoring=scorers, refit="acc", verbose=1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_

    # 6) Validation metrics
    y_val_pred = best.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_prec = precision_score(y_val, y_val_pred, average="weighted", zero_division=0)
    val_rec = recall_score(y_val, y_val_pred, average="weighted", zero_division=0)

    # 7) Test metrics
    y_test_pred = best.predict(X_test)
    test_acc  = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, average="weighted", zero_division=0)
    test_rec  = recall_score(y_test, y_test_pred, average="weighted", zero_division=0)

    # 8) Learning curve (στο train set για το best pipeline)
    sizes, train_scores, val_scores = learning_curve(
        best, X_train, y_train, cv=5, n_jobs=n_jobs, scoring="accuracy",
        train_sizes=np.linspace(0.1, 1.0, 5), shuffle=True, random_state=random_state
    )
    plot_learning_curve(sizes, train_scores, val_scores, out_dir / "learning_curves" / "svm_learning_curve.png")

    # 9) Confusion matrix
    plot_confusion(y_test, y_test_pred, labels=list(range(len(class_names))),
                   out_path=out_dir / "confusion_matrices" / "svm_confusion_matrix.png")

    # 10) Reports + JSON
    report_val  = classification_report_str(y_val,  y_val_pred, labels=list(range(len(class_names))))
    report_test = classification_report_str(y_test, y_test_pred, labels=list(range(len(class_names))))
    save_text(report_val,  out_dir / "svm_validation_report.txt")
    save_text(report_test, out_dir / "svm_test_report.txt")

    metrics = {
        "best_params": gs.best_params_,
        "validation": {"accuracy": val_acc, "precision_w": val_prec, "recall_w": val_rec},
        "test":       {"accuracy": test_acc, "precision_w": test_prec, "recall_w": test_rec},
        "labels": class_names
    }
    save_json(metrics, out_dir / "svm_metrics.json")
    print("Done. Metrics written to:", out_dir)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data", help="folder with train.txt/dev.txt/test.txt")
    ap.add_argument("--out_dir", type=str, default="results", help="output folder")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    run(args.data_dir, args.out_dir, n_jobs=args.n_jobs, random_state=args.seed)
