# 🫀 CardioCare AI — Machine Learning Heart Disease Risk Predictor

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-brightgreen.svg)](https://www.python.org/)
[![Model Accuracy](https://img.shields.io/badge/Model_Accuracy-95.83%25-cyan.svg)](#-model-benchmarks)
[![ROC-AUC Score](https://img.shields.io/badge/ROC--AUC-0.9943-purple.svg)](#-model-benchmarks)
[![SEO Optimized](https://img.shields.io/badge/SEO-Optimized-emerald.svg)](#-seo-optimizations)

**CardioCare AI** is an end-to-end, production-grade Machine Learning Mini Project designed to assess cardiovascular heart disease risk using 13 clinical diagnostic features. The system features a custom pure-NumPy supervised classification model, a Flask REST API backend, and an interactive glassmorphic web UI with explainable AI feature attribution and built-in SEO optimizations.

---

## 📌 Project Overview

- **Domain**: Healthcare & Clinical Decision Support Systems
- **ML Task**: Binary Classification (Heart Disease Presence vs. Absence)
- **Dataset**: 1,200 clinical patient samples with 13 UCI-standard diagnostic attributes
- **Model Accuracy**: **95.83%** on independent test set
- **ROC-AUC Score**: **0.9943**
- **Explainability**: Quantified feature importance contribution per prediction

---

## 📁 Repository Structure

```text
mini-project1/
├── index.html                  # SEO-optimized frontend UI with Schema.org JSON-LD
├── requirements.txt            # Python dependencies (NumPy, Flask, Flask-CORS)
├── .gitignore                  # Git ignore specifications
├── data/
│   ├── generate_dataset.py     # Reproducible clinical dataset generator
│   └── heart_disease_data.csv  # Synthetic dataset (1,200 records)
├── model/
│   ├── train.py                # Pure NumPy training, Z-score scaling & evaluation script
│   ├── heart_disease_model.json# Trained model weights & bias artifact
│   ├── scaler.json             # Feature Z-score mean & std scaling parameters
│   └── metrics.json            # Model evaluation metrics & feature importances
├── backend/
│   └── app.py                  # Flask REST API server (/api/predict, /api/metrics, /api/presets)
└── static/
    ├── css/
    │   └── style.css           # Glassmorphism design system & responsive layout
    └── js/
        └── app.js              # Live UI interaction, radial gauge & API fallback logic
```

---

## 📊 Model Benchmarks & Performance

Evaluated on 240 unseen test validation patient records:

| Metric | Score |
| :--- | :--- |
| **Accuracy** | **95.83%** |
| **ROC-AUC** | **0.9943** |
| **F1-Score** | **0.9716** |
| **Precision** | **95.77%** |
| **Recall** | **98.55%** |

---

## 🧠 Clinical Features

1. `age`: Patient Age in years
2. `sex`: Biological Sex (1 = Male, 0 = Female)
3. `cp`: Chest Pain Type (0 = Typical, 1 = Atypical, 2 = Non-anginal, 3 = Asymptomatic)
4. `trestbps`: Resting Blood Pressure (mm Hg)
5. `chol`: Serum Cholesterol (mg/dl)
6. `fbs`: Fasting Blood Sugar > 120 mg/dl (1 = True, 0 = False)
7. `restecg`: Resting ECG Results (0 = Normal, 1 = ST-T Abnormality, 2 = LV Hypertrophy)
8. `thalach`: Maximum Heart Rate Achieved
9. `exang`: Exercise Induced Angina (1 = Yes, 0 = No)
10. `oldpeak`: ST Depression Induced by Exercise Relative to Rest
11. `slope`: Peak Exercise ST Segment Slope
12. `ca`: Number of Major Vessels Colored by Fluoroscopy (0-3)
13. `thal`: Thalassemia Type (1 = Normal, 2 = Fixed Defect, 3 = Reversible Defect)

---

## 🚀 Quick Start Guide

### 1. Run Model Training Pipeline
To generate the dataset and train the model from scratch:

```bash
python data/generate_dataset.py
python model/train.py
```

### 2. Start Flask Backend Server
Launch the local REST API server:

```bash
python backend/app.py
```

Navigate to `http://localhost:5000` in your browser to interact with the system.

---

## 🔌 REST API Endpoints

### `POST /api/predict`
Calculates risk score percentage, severity tier, top contributing drivers, and clinical guidelines.

```json
{
  "age": 54, "sex": 1, "cp": 1, "trestbps": 138, "chol": 242,
  "fbs": 0, "restecg": 1, "thalach": 142, "exang": 0,
  "oldpeak": 1.4, "slope": 1, "ca": 1, "thal": 2
}
```

### `GET /api/metrics`
Returns complete model performance benchmarks and feature importance weights.

### `GET /api/presets`
Returns predefined clinical patient profile presets (Low, Moderate, High Risk).

---

## 🔍 SEO & Web Optimization Features

The web frontend automatically includes modern SEO best practices:
- **Semantic HTML5 Structure**: Structured with `<header>`, `<main>`, `<section>`, `<article>`, and `<footer>` elements.
- **Meta Title & Description**: Optimised for search engines and clinical query indexing.
- **Social Media Cards**: Complete Open Graph (`og:*`) and Twitter Card (`twitter:*`) tag integration.
- **Schema.org Structured Data**: Embedded `SoftwareApplication` JSON-LD markup.
- **Accessibility & UX**: Accessible color contrast, custom focus states, and responsive breakpoint grid.

---

## 📜 License

Distributed under the MIT License.
