"""
Extract features using a ViT-Base pretrained with DINO on ImageNet.

This tests whether the ViT architecture and self-supervised training
method account for performance differences, independent of pathology-
specific training data.

Usage:
    python extract_features_dino.py
"""

import torch
import timm
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from timm.data import resolve_data_config, create_transform
from tqdm import tqdm

# Use all available CPU cores for PyTorch operations
torch.set_num_threads(torch.get_num_threads())
torch.set_num_interop_threads(1)


def main():
    data_dir = Path("data")
    out_dir = Path("embeddings")
    out_dir.mkdir(exist_ok=True)

    # vit_base_patch16_224.dino is publicly available through timm
    # no Hugging Face token needed
    print("Loading DINO ViT-Base...")
    model = timm.create_model(
        "vit_base_patch16_224.dino",
        pretrained=True,
        num_classes=0,
    )
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"DINO ViT-Base parameters: {param_count:,}")

    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config)

    # Load annotations
    df = pd.read_csv(data_dir / "annotations.csv")
    image_names = df["Image Name"].tolist()
    print(f"Found {len(image_names)} images")

    # Extract features in batches
    image_dir = data_dir / "images"
    features = []
    batch_size = 16

    with torch.no_grad():
        for i in tqdm(range(0, len(image_names), batch_size),
                      desc="Extracting DINO features"):
            batch_names = image_names[i : i + batch_size]
            tensors = []
            for name in batch_names:
                img = Image.open(image_dir / name).convert("RGB")
                tensors.append(transform(img))

            batch = torch.stack(tensors)
            feats = model(batch)
            features.append(feats.numpy())

    features = np.vstack(features)
    print(f"Feature matrix shape: {features.shape}")

    out_path = out_dir / "dino_vitb16.npz"
    np.savez_compressed(
        out_path,
        features=features,
        image_names=np.array(image_names),
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()