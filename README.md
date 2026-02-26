---
title: ClearAudit
emoji: 🦅
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
app_port: 7860
---

# ClearAudit VN: Real-Time ML Fraud Detection Ecosystem

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**An institutional-grade fraud detection system specifically architected for the Vietnamese financial market.** ClearAudit VN combines high-fidelity transaction simulation, generative data augmentation, and a dual-branch ensemble architecture to detect sophisticated fraud typologies with **99.7% Recall** at sub-150ms latency.

---

## 🚀 The Vision

Legacy rule-based systems in Vietnam often fail to catch regional-specific threats like **Fake Shipper COD** or **QR Quishing (Phishing)** while generating excessive false positives. ClearAudit VN solves this by treating fraud detection as a localized, end-to-end engineering challenge rather than a generic classification task.

### Key Performance Benchmarks
*   **99.7% Recall** across core Vietnamese fraud typologies.
*   **< 12ms P95 Feature Hydration** using Redis and DuckDB.
*   **SBV-Compliant** Explainability (SHAP).
*   **Real-time Scoring** serving 550,000+ transaction patterns.

---

## 🛠️ Technical Stack

-   **Core**: Python 3.12, FastAPI
-   **Machine Learning**: XGBoost (Supervised), Keras/TensorFlow (Autoencoder), Scikit-Learn
-   **Data Eng & Augmentation**: CTGAN, WGAN-GP, DuckDB, Pandas
-   **Infrastructure & MLOps**: Redis (Live State), SHAP (Explainability), Evidently AI (Monitoring)
-   **Frontend**: Vanilla JS, GSAP, CSS3 (Glassmorphism & Micro-animations)

---

## 🧠 System Architecture: The Dual-Branch Approach

ClearAudit employs a unique ensemble strategy to ensure both precision against known threats and safety against "zero-day" attacks.

1.  **The Supervised Branch (XGBoost)**: Trained on 550,000 synthetic transactions. It excels at identifying high-precision fraud signatures such as **Biometric Bypass** and **Account Takeover**.
2.  **The Unsupervised Branch (Autoencoder)**: Monitors for anomalies based on reconstruction error (MSE). It flags transactions that "don't look right," providing a secondary safety net for emerging fraud patterns not yet present in the training data.

---

## 📈 The End-to-End Pipeline

### 1. SimPay VN Simulation
Since real banking PII is protected, we built **SimPay VN**, a proprietary generator that simulates:
-   9 major Vietnamese banks and e-wallets (VCB, TCB, MoMo, etc.).
-   Realistic VND distributions and regional temporal spending habits.
-   4 key fraud typologies: Fake Shipper, Quishing, Biometric Evasion, and ATO.

### 2. Generative Data Augmentation
Using **CTGAN** and **WGAN-GP**, we scaled the initial seed data into a robust training set of 550,000 samples, ensuring the model generalizes well to complex, multi-modal distributions.

### 3. Feature Engineering (The DuckDB Engine)
We engineer 32 real-time features including:
-   **Velocity**: Rolling 1h/24h transaction windows.
-   **Temporal**: Cyclical hour/day encoding for Tết and holiday peaks.
-   **Graph**: Repeat pair frequency and sender-receiver degree centrality.

### 4. Precision Calibration (SMOTE)
Through extensive experimentation, we validated that a **three-layer balancing strategy** (CTGAN + mild SMOTE + Weighted Loss) improved precision by **10.7 points**, resulting in a highly stable decision boundary.

---

## ⚡ Getting Started

The project is orchestrated via a centralized **Control Hub** for easy local exploration.

### 1. Prerequisites
- Python 3.10+
- Node.js (for `npx http-server`)
- Redis Server (local or cloud)

### 2. Installation
```bash
git clone https://github.com/ShyamNayak27/ClearAudit.git
cd ClearAudit
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### 3. Launch the Control Hub
Run the centralized orchestrator to start the API, Landing Page, Technical Portfolio, and Service Dashboard:
```bash
python project_hub.py
```

---

## 📑 Documentation

For a deep dive into the engineering rationale and experimental results:
-   **[Live Demo (Hugging Face)](https://huggingface.co/spaces/KazeSensei/ClearAudit)**: Interact with the production system.
-   **[Technical Whitepaper (PDF)](frontend/technical_page/Documentation.pdf)**: Detailed methodology and benchmarks.
-   [Phase 1-3 Complete Deep-Dive (Markdown)](docs/complete_technical_deep_dive.md)
-   [SMOTE vs No-SMOTE Experiment Report](docs/smote_experiment_report.md)

---

## 👨‍💻 About the Creator
**Shyam Narayan Nayak**
Machine Learning Systems & Backend Infrastructure Engineer.
*   **LinkedIn**: [shyamnnayak](https://www.linkedin.com/in/shyamnnayak/)
*   **GitHub**: [ShyamNayak27](https://github.com/ShyamNayak27)


---
*ClearAudit VN is an ML Engineering Portfolio project. 2026 MIT License.*
