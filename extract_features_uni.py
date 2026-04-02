"""
Extract feature embeddings from MHIST images using the UNI foundation model.

UNI (Chen et al., 2024) is a pathology-specific ViT-Large pretrained on
100M+ histopathology patches via DINOv2 self-supervised learning.

Usage:
    python extract_features_uni.py
"""
\

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

    # ── Load UNI ──
    # timm.create_model can pull directly from a Hugging Face Hub repo
    # by using the "hf_hub:" prefix. This downloads the weights on
    # first run and caches them locally for future use.
    print("Loading UNI model (this may take a few minutes on first run)...")

    # UNI uses a ViT-Large with LayerScale (init_values > 0 enables ls1/ls2 params).
    # We create the architecture with the right config first, then load weights.
    model = timm.create_model(
        "vit_large_patch16_224",
        init_values=1e-5,   # enables LayerScale layers (ls1.gamma, ls2.gamma)
        num_classes=0,
        pretrained=False,   # don't load ImageNet weights
    )

    # Download and load UNI weights from Hugging Face
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(
        repo_id="MahmoodLab/uni",
        filename="pytorch_model.bin",
    )
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    # UNI is a ViT-Large — its feature dimension is 1024
    # (compared to ResNet50's 2048)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"UNI parameters: {param_count:,}")

    # Get UNI's expected preprocessing
    config = resolve_data_config(model.pretrained_cfg)
    transform = create_transform(**config)

    # ── Load annotations ──
    df = pd.read_csv(data_dir / "annotations.csv")
    image_names = df["Image Name"].tolist()
    print(f"Found {len(image_names)} images")

    # ── Extract features in batches ──
    image_dir = data_dir / "images"
    features = []
    batch_size = 16  # process 16 images at once

    with torch.no_grad():
        for i in tqdm(range(0, len(image_names), batch_size),
                      desc="Extracting UNI features"):
            batch_names = image_names[i : i + batch_size]
            tensors = []
            for name in batch_names:
                img = Image.open(image_dir / name).convert("RGB")
                tensors.append(transform(img))

            # torch.stack combines individual tensors into one batch
            # shape goes from list of [C,H,W] to [batch_size, C, H, W]
            batch = torch.stack(tensors)
            feats = model(batch)
            features.append(feats.numpy())

    features = np.vstack(features)

    # ── Save ──
    out_path = out_dir / "uni.npz"
    np.savez_compressed(
        out_path,
        features=features,
        image_names=np.array(image_names),
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()