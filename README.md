# Amazon_Smart_Product_Pricing_Challenge 🚀

---

## 📌 Overview

This project presents a **multimodal deep learning solution** for the **Smart Product Pricing Challenge**, aimed at accurately predicting product prices by jointly analyzing **textual descriptions**, **product images**, and an **engineered numerical feature**.

The system leverages state-of-the-art pretrained models for both Natural Language Processing (NLP) and Computer Vision (CV), combined with domain-aware feature engineering and robust training strategies.

---

## 🧠 Methodology

### Problem Insight

Exploratory Data Analysis (EDA) revealed that product prices are **heavily right-skewed**, with significant outliers. Direct regression on raw prices resulted in unstable training and poor generalization.

### Target Transformation

To address this, the following preprocessing was applied to the target variable:

- **Outlier Clipping:** Prices clipped between the **1st and 99th percentiles**
- **Log Transformation:**  
  \[
  y = \log(1 + price)
  \]

The model is trained to predict **log(price)**, and predictions are converted back using:

\[
price = \exp(y) - 1
\]

---

## 🏗️ Model Architecture

The architecture is a **custom multimodal neural network** implemented in PyTorch.

### 1. Text Encoder
- Model: **`distilroberta-base`**
- Input: `catalog_content`
- Output: CLS token embedding
- Rationale: Lightweight yet powerful language representation

### 2. Image Encoder
- Model: **`efficientnet_b0`**
- Input: Product image
- Output: High-level visual feature vector
- Rationale: Strong performance with low computational cost

### 3. Feature Fusion & Regression Head
- Inputs:
  - Text embedding
  - Image embedding
  - Item Pack Quantity (IPQ)
- Architecture:
  - LayerNorm
  - Fully Connected Layers
  - ReLU Activation
  - Dropout
- Output: Single scalar value representing **log(price)**

---

## 🧪 Feature Engineering

### Item Pack Quantity (IPQ)

A custom regex-based parser extracts pack quantity information directly from the product description.

Examples:
- `"Pack of 24"` → `24`
- `"12 count"` → `12`
- No quantity found → `1`

This feature improves pricing accuracy by explicitly modeling quantity-based price scaling.

---

## 🎯 Training Strategy & Regularization

To prevent overfitting and improve generalization, the following techniques were applied:

- **Loss Function:** Mean Squared Error (MSE) on log-transformed prices
- **Optimizer:** AdamW
- **Weight Decay:** `0.01`
- **Image Augmentation:**
  - RandomHorizontalFlip
  - ColorJitter
- **Dropout:** `0.2` in regression head
- **Early Stopping:**
  - Validation metric: **SMAPE**
  - Training stopped if no improvement for 2 consecutive epochs

---

## 📁 Project Structure

smart_pricing/
│
├── best_model.pth
│
├── dataset/
│ ├── train.csv
│ ├── test.csv
│ ├── train_images/
│ └── test_images/
│
└── src/
├── config.py
├── dataset.py
├── model.py
└── inference.py


---

## ⚙️ Configuration

All paths and hyperparameters are centrally managed via `config.py`.

Key parameters include:
- Batch size
- Image size
- Maximum text length
- Pretrained model selection
- Device configuration (CPU / GPU)

---

## ▶️ Inference Pipeline

The inference workflow is fully automated:

1. Load `test.csv`
2. Tokenize product text
3. Load and preprocess product images
4. Load trained model weights
5. Predict log(price)
6. Convert predictions back to original scale
7. Save results to `submission.csv`

---

## ▶️ How to Run Inference

From the `src` directory:

```bash
python inference.py

