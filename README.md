# Explainable AI Intrusion Detection System for IoT Networks

An explainable deep learning system for real-time intrusion detection in IoT network traffic, developed as part of an MSc dissertation in Artificial Intelligence and Data Science at the University of Hull.

---

## Overview

This project addresses a core challenge in IoT security: building an intrusion detection system that is not only accurate but also interpretable. Standard black-box models may achieve high accuracy but offer no insight into why a prediction was made — which matters in security contexts where trust and auditability are critical.

The solution combines a hybrid deep learning architecture with post-hoc explainability tools (SHAP and LIME) to produce a system that detects network intrusions reliably and can explain its decisions in a human-readable way.

---

## Architecture

The final model is a **CNN + BiLSTM + Multi-Head Self-Attention** architecture, selected through a five-model ablation study that progressively introduced components to isolate the contribution of each:

| Model | Architecture |
|---|---|
| M1 | CNN + LSTM (baseline) |
| M2 | CNN + LSTM + training optimisations |
| M3 | CNN + BiLSTM |
| M4 | CNN + BiLSTM + Single-Head Attention |
| M5 | CNN + BiLSTM + Multi-Head Self-Attention (final) |

Each model was trained and evaluated under identical conditions to ensure fair comparison. SMOTE was applied to address class imbalance in the dataset.

---

## Dataset

The project uses the **CIC-BoT-IoT dataset**, a widely used benchmark for IoT network intrusion detection containing labelled network traffic flows across multiple attack categories.

---

## Explainability

Two post-hoc explainability methods are integrated:

- **SHAP (SHapley Additive exPlanations)** — used to identify which features contribute most to individual predictions and to surface global feature importance across the dataset
- **LIME (Local Interpretable Model-agnostic Explanations)** — used to generate local explanations for individual classification decisions

Both methods are applied to the final M5 model and visualised in the accompanying notebooks.

---

## Repository Structure

```
├── .devcontainer/         # Dev container configuration
├── results/               # Model evaluation outputs and ablation results
├── 02_preprocessing.ipynb # Data preprocessing and feature engineering
├── Untitled.ipynb         # Model training and evaluation notebooks
├── app.py                 # Application entry point
├── CICFlowMeter Features.csv  # Feature definitions for the dataset
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Tech Stack

- **Python** — TensorFlow/Keras, scikit-learn, NumPy, Pandas
- **Explainability** — SHAP, LIME
- **Class balancing** — imbalanced-learn (SMOTE)
- **Evaluation** — precision, recall, F1-score, ROC-AUC, confusion matrix

---

## Usage

**Install dependencies:**
```bash
git clone https://github.com/Ebube22/xai-ids-iot
cd xai-ids-iot
pip install -r requirements.txt
```

**Run preprocessing:**
Open and run `02_preprocessing.ipynb` to preprocess the CIC-BoT-IoT dataset and generate ML-ready features.

**Train models and view results:**
Open `Untitled.ipynb` for the full model training, ablation study, and SHAP/LIME evaluation pipeline.

**Run the application:**
```bash
python app.py
```

---

## Key Concepts

- **Ablation study** — a systematic approach to understanding model performance by isolating the contribution of each architectural component
- **Multi-Head Self-Attention** — a transformer-inspired mechanism that allows the model to attend to different parts of a sequence simultaneously, improving detection of complex attack patterns
- **BiLSTM** — a bidirectional LSTM that processes sequences in both directions, capturing temporal dependencies in network traffic more effectively than a unidirectional LSTM
- **SMOTE** — Synthetic Minority Oversampling Technique, used to handle class imbalance between normal and attack traffic

---

## Academic Context

This project was submitted as the dissertation component of an MSc in Artificial Intelligence and Data Science at the University of Hull (graduating May 2026), supervised by Dr. Funmilola.

---

## License

MIT License
