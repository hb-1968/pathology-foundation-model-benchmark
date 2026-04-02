# Pathology Foundation Model Benchmark

Comparing pretrained foundation model representations for histopathology 
image classification on the MHIST dataset.

## Models Compared
- **ResNet-50** (ImageNet pretrained) — general-purpose baseline
- **UNI** (Chen et al., 2024) — pathology-specific foundation model
- **CONCH** (Lu et al., 2024) — vision-language pathology model

## Dataset
[MHIST](https://bmirds.github.io/MHIST/) — 3,152 histopathology images 
(224×224) for binary classification of colorectal polyps (HP vs. SSA).

## Method
Feature extraction with frozen pretrained encoders → linear probe 
(logistic regression) → ROC/AUC comparison.

## Setup
```bash
conda create -n pathbench python=3.11 -y
conda activate pathbench
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install scikit-learn matplotlib pandas pillow tqdm huggingface_hub timm
```

## Results
*(To be added)*
```