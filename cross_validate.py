"""
Cross-validated comparison of all models on MHIST.

Uses stratified 5-fold CV on the training set to produce
mean and standard deviation for accuracy and AUC.

Usage:
    python cross_validate.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt


def evaluate_model(model_name, df, emb_dir, n_folds=5):
    """Run stratified k-fold CV and return per-fold metrics."""
    data = np.load(emb_dir / f"{model_name}.npz")
    features = data["features"]
    image_names = data["image_names"]

    name_to_idx = {n: i for i, n in enumerate(image_names)}

    # Use only the training partition for cross-validation
    train_df = df[df["Partition"] == "train"]
    idxs = [name_to_idx[n] for n in train_df["Image Name"]]
    X = features[idxs]
    y = (train_df["Majority Vote Label"] == "SSA").astype(int).values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    accs = []
    aucs = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)
        y_prob = clf.predict_proba(X_val)[:, 1]

        accs.append(accuracy_score(y_val, y_pred))
        aucs.append(roc_auc_score(y_val, y_prob))

    return np.array(accs), np.array(aucs)


def main():
    data_dir = Path("data")
    emb_dir = Path("embeddings")
    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    df = pd.read_csv(data_dir / "annotations.csv")

    models = {
        "ResNet-50\n(ImageNet)": "resnet50",
        "DINO ViT-B\n(ImageNet)": "dino_vitb16",
        "UNI ViT-L\n(Pathology)": "uni",
    }

    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    # ── Run cross-validation ──
    all_results = []

    for label, model_name in models.items():
        accs, aucs = evaluate_model(model_name, df, emb_dir)
        print(f"{label.replace(chr(10), ' ')}")
        print(f"  Accuracy: {accs.mean():.4f} ± {accs.std():.4f}")
        print(f"  AUC:      {aucs.mean():.4f} ± {aucs.std():.4f}")
        print()
        all_results.append({
            "label": label,
            "model": model_name,
            "acc_mean": accs.mean(),
            "acc_std": accs.std(),
            "auc_mean": aucs.mean(),
            "auc_std": aucs.std(),
            "fold_accs": accs,
            "fold_aucs": aucs,
        })

    # ── Bar chart with error bars ──
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    labels = [r["label"] for r in all_results]
    acc_means = [r["acc_mean"] for r in all_results]
    acc_stds = [r["acc_std"] for r in all_results]
    auc_means = [r["auc_mean"] for r in all_results]
    auc_stds = [r["auc_std"] for r in all_results]

    x = range(len(labels))

    axes[0].bar(x, acc_means, yerr=acc_stds, color=colors,
                edgecolor="black", linewidth=0.5, capsize=5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=10)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_ylim(0.70, 0.88)
    axes[0].set_title("Accuracy (5-Fold CV)", fontsize=13)

    axes[1].bar(x, auc_means, yerr=auc_stds, color=colors,
                edgecolor="black", linewidth=0.5, capsize=5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=10)
    axes[1].set_ylabel("AUC", fontsize=12)
    axes[1].set_ylim(0.80, 0.92)
    axes[1].set_title("AUC (5-Fold CV)", fontsize=13)

    fig.suptitle("Model Comparison: 5-Fold Cross-Validation on MHIST",
                 fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "cv_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved to {out_dir / 'cv_comparison.png'}")

    # ── Save CV results to CSV ──
    cv_df = pd.DataFrame([{
        "model": r["model"],
        "acc_mean": r["acc_mean"],
        "acc_std": r["acc_std"],
        "auc_mean": r["auc_mean"],
        "auc_std": r["auc_std"],
    } for r in all_results])
    cv_df.to_csv(out_dir / "cv_results.csv", index=False)
    print(f"Saved to {out_dir / 'cv_results.csv'}")


if __name__ == "__main__":
    main()