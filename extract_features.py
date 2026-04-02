"""
Extract feature embeddings from MHIST images using a pretrained model.

Usage:
    python extract_features.py --model resnet50
"""

import argparse
import torch
import timm
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm


def load_model(model_name):
    """
    Load a pretrained model with its classification head removed.
    
    num_classes=0 strips the final linear layer so the model
    outputs its penultimate feature vector instead of class logits.
    """
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model.eval()

    # resolve_data_config reads the model's expected input format
    # (image size, normalization stats, interpolation method)
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config)

    return model, transform


def extract(model, transform, image_dir, image_names):
    """
    Run each image through the model and collect feature vectors.
    
    torch.no_grad() disables gradient tracking, which saves memory
    and speeds up inference since we're not training anything.
    """
    features = []

    with torch.no_grad():
        for name in tqdm(image_names, desc="Extracting features"):
            img_path = image_dir / name
            img = Image.open(img_path).convert("RGB")

            # transform converts PIL image -> normalized tensor
            # .unsqueeze(0) adds a batch dimension: [C, H, W] -> [1, C, H, W]
            tensor = transform(img).unsqueeze(0)

            feat = model(tensor)

            # .squeeze(0) removes batch dim, .numpy() converts to NumPy
            features.append(feat.squeeze(0).numpy())

    return np.stack(features)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="resnet50",
        help="timm model name (e.g., resnet50, vit_large_patch16_224)"
    )
    parser.add_argument("--data-dir", default="data", help="path to MHIST data")
    parser.add_argument("--out-dir", default="embeddings", help="where to save features")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True)

    # Load annotations
    df = pd.read_csv(data_dir / "annotations.csv")
    image_names = df["Image Name"].tolist()
    print(f"Found {len(image_names)} images")

    # Load model
    print(f"Loading model: {args.model}")
    model, transform = load_model(args.model)

    # Count parameters (informational — useful for the writeup)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Extract
    image_dir = data_dir / "images"
    features = extract(model, transform, image_dir, image_names)
    print(f"Feature matrix shape: {features.shape}")

    # Save
    out_path = out_dir / f"{args.model}.npz"
    np.savez_compressed(
        out_path,
        features=features,
        image_names=np.array(image_names),
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()