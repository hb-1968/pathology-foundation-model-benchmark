import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, accuracy_score, classification_report, roc_curve
)
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="resnet50",
        help="name matching the .npz file in embeddings/"
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--emb-dir", default="embeddings")
    parser.add_argument("--out-dir", default="results")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # ── Load annotations ──
    df = pd.read_csv(Path(args.data_dir) / "annotations.csv")

    # ── Load features ──
    data = np.load(Path(args.emb_dir) / f"{args.model}.npz")
    features = data["features"]
    image_names = data["image_names"]
    print(f"Loaded features: {features.shape}")

    # ── Align features with annotations ──
    # Build a lookup from image name to row index in the feature matrix
    name_to_idx = {name: i for i, name in enumerate(image_names)}

    # MHIST uses a "Partition" column to mark train vs test
    train_mask = df["Partition"] == "train"
    test_mask = df["Partition"] == "test"

    def gather(mask):
        """Pull feature rows and labels for a given split."""
        subset = df[mask]
        idxs = [name_to_idx[n] for n in subset["Image Name"]]
        X = features[idxs]
        # Convert string labels to binary: SSA=1, HP=0
        y = (subset["Majority Vote Label"] == "SSA").astype(int).values
        return X, y

    X_train, y_train = gather(train_mask)
    X_test, y_test = gather(test_mask)
    print(f"Train: {X_train.shape[0]} images  |  Test: {X_test.shape[0]} images")
    print(f"Train class balance: {y_train.mean():.2%} SSA")
    print(f"Test class balance:  {y_test.mean():.2%} SSA")

    # ── Train logistic regression ──
    # max_iter=1000 because high-dimensional features sometimes
    # need more iterations to converge than the default 100
    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_train, y_train)

    # ── Evaluate ──
    y_pred = clf.predict(X_test)

    # predict_proba returns [[p(HP), p(SSA)]] per sample
    # we take column 1 for the positive class probability
    y_prob = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"\n{'='*40}")
    print(f"Model: {args.model}")
    print(f"Accuracy: {acc:.4f}")
    print(f"AUC:      {auc:.4f}")
    print(f"{'='*40}")
    print(classification_report(y_test, y_pred, target_names=["HP", "SSA"]))

    # ── Save ROC curve ──
    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f"{args.model} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC = 0.500)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {args.model} Linear Probe on MHIST")
    ax.legend(loc="lower right")
    ax.set_aspect("equal")
    fig.tight_layout()

    fig_path = out_dir / f"roc_{args.model}.png"
    fig.savefig(fig_path, dpi=150)
    print(f"ROC curve saved to {fig_path}")

    # ── Save numeric results to CSV (for later comparison) ──
    results_path = out_dir / "results.csv"
    row = pd.DataFrame([{
        "model": args.model,
        "accuracy": acc,
        "auc": auc,
        "train_size": X_train.shape[0],
        "test_size": X_test.shape[0],
        "feature_dim": features.shape[1],
    }])

    if results_path.exists():
        existing = pd.read_csv(results_path)
        # Replace row if model already exists, otherwise append
        existing = existing[existing["model"] != args.model]
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row

    combined.to_csv(results_path, index=False)
    print(f"Results appended to {results_path}")


if __name__ == "__main__":
    main()