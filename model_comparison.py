"""
Generate comparison figures across all evaluated models.

Usage:
    python compare_models.py
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score


def main():
    data_dir = Path("data")
    emb_dir = Path("embeddings")
    out_dir = Path("results")

    df = pd.read_csv(data_dir / "annotations.csv")
    test_mask = df["Partition"] == "test"
    train_mask = df["Partition"] == "train"

    models = {
        "ResNet-50 (ImageNet)": "resnet50",
        "DINO ViT-B/16 (ImageNet)": "dino_vitb16",
        "UNI ViT-L/16 (Pathology)": "uni",
    }

    # ── ROC Comparison ──
    fig, ax = plt.subplots(figsize=(7, 7))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    for (label, model_name), color in zip(models.items(), colors):
        data = np.load(emb_dir / f"{model_name}.npz")
        features = data["features"]
        image_names = data["image_names"]

        name_to_idx = {n: i for i, n in enumerate(image_names)}

        def gather(mask):
            subset = df[mask]
            idxs = [name_to_idx[n] for n in subset["Image Name"]]
            X = features[idxs]
            y = (subset["Majority Vote Label"] == "SSA").astype(int).values
            return X, y

        X_train, y_train = gather(train_mask)
        X_test, y_test = gather(test_mask)

        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)

        ax.plot(fpr, tpr, linewidth=2, color=color,
                label=f"{label} (AUC = {auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Comparison: Linear Probe on MHIST", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_comparison.png", dpi=150)
    print(f"Saved ROC comparison to {out_dir / 'roc_comparison.png'}")

    # ── Bar Chart ──
    results = pd.read_csv(out_dir / "results.csv")

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Map model file names to display labels
    display_names = {
        "resnet50": "ResNet-50\n(ImageNet)",
        "dino_vitb16": "DINO ViT-B\n(ImageNet)",
        "uni": "UNI ViT-L\n(Pathology)",
    }

    names = [display_names[m] for m in results["model"]]

    axes[0].bar(names, results["accuracy"], color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_ylabel("Accuracy", fontsize=12)
    axes[0].set_ylim(0.70, 0.85)
    axes[0].set_title("Accuracy", fontsize=13)

    axes[1].bar(names, results["auc"], color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_ylabel("AUC", fontsize=12)
    axes[1].set_ylim(0.80, 0.90)
    axes[1].set_title("AUC (ROC)", fontsize=13)

    fig.suptitle("Model Comparison: Linear Probe on MHIST", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(out_dir / "bar_comparison.png", dpi=150, bbox_inches="tight")
    print(f"Saved bar comparison to {out_dir / 'bar_comparison.png'}")


if __name__ == "__main__":
    main()