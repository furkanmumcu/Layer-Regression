# Layer Regressin

[![Paper](https://img.shields.io/badge/Paper-TMLR_2025-b31b1b.svg)](https://openreview.net/forum?id=OCY5APFnFI)

Official implementation of the paper **"Universal and Efficient Detection of Adversarial Data through Nonuniform Impact on Network Layers"** (Transactions on Machine Learning Research, 2025).

Layer Regression (LR) is a lightweight defense method that detects adversarial attacks by monitoring the consistency between early-layer and deep-layer features.

---

## 🚀 Key Highlights

- **Universal:** Compatible with any architecture (CNNs, Transformers) and domain (Image, Video, Audio).
- **Computationally Efficient:** Orders of magnitude faster than state-of-the-art detectors (e.g., EPS-AD, VLAD).
- **Real-Time:** Adds negligible latency (~0.0004s per sample), suitable for resource‑constrained systems.
- **Automated Setup:** Automatically discovers and selects optimal network layers for detection (Algorithm 1).

---

## 🛠️ Installation

```bash
git clone https://github.com/furkanmumcu/Layer-Regression.git
cd Layer-Regression
pip install torch torchvision timm numpy
```

[Download](https://drive.google.com/file/d/10FQEQ1yOT9Jtd7hESfDbYcT2cqa-Eg5t) and place the validation set of the 2012 ImageNet (ILSVRC 2012) in the directory `/dataset/imagenet/val`.

---

## ⚡ Quick Start

### **1. Training**

The training script automatically analyzes the target model (such as ResNet, Inception, or ViT), selects appropriate layers using the method described in the paper, and trains the lightweight detector.

Run:

```bash
python train.py
```

A checkpoint named `lr_detector_{model_name}.pt` will be saved, containing both the model weights and the selected layer configuration.

---

### **2. Testing**

To evaluate the detector against adversarial attacks, run:

```bash
python test.py
```

The testing script restores the configuration used during training to ensure compatibility.

---

## 🧠 Method Overview

Deep neural networks process information layer by layer. Although adversarial perturbations are nearly imperceptible in the input space, their impact grows as the data propagates through deeper layers. This increasing discrepancy forms a measurable “disconnect” between early- and late-layer feature representations.

Layer Regression leverages this nonuniform impact:

1. **Early-layer features:** Extract feature vectors from shallow layers, where adversarial effects remain minimal.
2. **Prediction:** A lightweight MLP predicts the corresponding deep-layer features expected for clean inputs.
3. **Detection:**  
   - Clean inputs → accurate deep‑feature predictions  
   - Adversarial inputs → distorted deep features → high MSE → detection flag  

LR requires no modifications to the target model and does not require adversarial training.

---

## 📝 Citation

```bibtex
@article{mumcu2025universal,
  title={Universal and Efficient Detection of Adversarial Data through Nonuniform Impact on Network Layers},
  author={Mumcu, Furkan and Yilmaz, Yasin},
  journal={arXiv preprint arXiv:2506.20816},
  year={2025}
}

```

---
