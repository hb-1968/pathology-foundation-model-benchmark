import torch
import timm
import pandas as pd
from PIL import Image
from pathlib import Path

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"timm version: {timm.__version__}")

# Check data
annotations = Path("data/annotations.csv")
if annotations.exists():
    df = pd.read_csv(annotations)
    print(f"Annotations loaded: {len(df)} images")
    print(f"Columns: {list(df.columns)}")
    print(f"Label distribution:\n{df['Majority Vote Label'].value_counts()}")
else:
    print("annotations.csv not found — download MHIST first")

# Check that a ResNet50 loads (our baseline model)
model = timm.create_model("resnet50", pretrained=True, num_classes=0)
model.eval()
dummy = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    features = model(dummy)
print(f"ResNet50 feature dim: {features.shape[1]}")
print("\nSetup complete!")