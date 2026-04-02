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
```
*ResNet-50 Results:*
Train: 2175 images  |  Test: 977 images
Train class balance: 28.97% SSA
Test class balance:  36.85% SSA

========================================
Model: resnet50
Accuracy: 0.7963
AUC:      0.8671
========================================
              precision    recall  f1-score   support

          HP       0.82      0.86      0.84       617
         SSA       0.74      0.69      0.71       360

    accuracy                           0.80       977
   macro avg       0.78      0.77      0.78       977
weighted avg       0.79      0.80      0.79       977

```