# On the Role of Learning Rate Warmup in Small Networks Trained with SGD

A controlled ablation of learning rate warmup in a 3‑layer MLP trained with plain SGD on MNIST and Fashion-MNIST. This repository contains all code, experimental logs, and analysis scripts to reproduce the results of the paper.

> **Paper:** *On the Role of Learning Rate Warmup in Small Networks Trained with SGD*  
> **Authors:** ([Nikhil Aaryan Singh](mailto:nikhilaaryans@gmail.com))


---

## Key Results

- No statistically significant difference between warmup strategies (Welch’s t-test, α = 0.05)
- Final validation accuracy:
  - MNIST: ~97.7%
  - Fashion-MNIST: ~89.1–89.2%
- Warmup does not meaningfully improve:
  - Convergence speed
  - Final performance
  - Generalization gap

---

## Why This Matters

Learning rate warmup is widely used in large-scale deep learning, but its necessity in smaller models is often assumed rather than tested. This work provides empirical evidence that challenges its importance in low-capacity regimes trained with standard SGD.

---

## Overview

We compare three learning rate schedules:
- **No warmup** – constant LR = 0.01
- **Linear warmup** – LR ramps from 0 to 0.01 over 5 epochs
- **Cosine warmup** – half‑cosine ramp from 0 to 0.01 over 5 epochs

**Training setup:**
- Optimizer: SGD (momentum = 0.9)
- Batch size: 64
- Epochs: 30
- Architecture: MLP (784 → 256 → 128 → 10)
- Seeds: 10 per condition (total = 60 runs)

---

## Repository Structure

```
warmup-sgd/
├── README.md
├── requirements.txt
├── train.py                # Single training run
├── run_all.py              # Run all 60 experiments
├── analysis.ipynb          # Full analysis (tables + figures)
├── results/
│   ├── mnist_none_seed0.csv
│   ├── ...
│   └── fmnist_cosine_seed9.csv
└── figures/
    ├── figure1_epochs_to_threshold.png
    ├── figure2_gradient_norm.png
    └── figure3_generalization_gap.png
```


---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/nikhilaaryans/warmup-sgd.git
cd warmup-sgd
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- torch
- torchvision
- numpy
- pandas
- scipy
- matplotlib
- seaborn

---

## Reproducing the Experiments

### Run a Single Configuration

```bash
python train.py --dataset mnist --scheduler linear --seed 0
```

This will:
- Download dataset (if needed)
- Train for 30 epochs
- Save results to:
  ```
  results/mnist_linear_seed0.csv
  ```

---

### Run All Experiments

```bash
python run_all.py
```

**Expected output:**
- 60 CSV files inside `results/`

---

## Reproducing Analysis & Figures

1. Open `analysis.ipynb` (Jupyter / Colab)

2. Set:
```python
RESULTS_DIR = "results/"
```

3. Run all cells

This will:
- Compute mean ± std for final accuracy
- Perform pairwise Welch’s t-tests (Table 1)
- Generate:
  - Figure 1: Epochs to threshold
  - Figure 2: Gradient norm dynamics
  - Figure 3: Generalization gap
- Save figures to `figures/`

---

## Reproducibility

- 10 independent seeds per condition  
- Fixed architecture and optimizer  
- Controlled hyperparameters across all runs  
- Statistical testing via Welch’s t-test  
- Full raw logs available in `results/`  