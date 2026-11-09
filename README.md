# 🛡️ Sentinel-AI: Explainable Intrusion Detection

Sentinel-AI is a real-time Network Intrusion Detection System (NIDS) that identifies and explains cyber threats using Deep Learning.

### 🚀 [Live Dashboard on Hugging Face](https://huggingface.co/spaces/xomarouane/sentinel-ai-cybersecurity)

---

## 🧠 Project Overview
Modern cybersecurity often suffers from "black-box" AI—where a model detects a threat but cannot explain why. Sentinel-AI solves this by integrating **Explainable AI (XAI)** directly into the security workflow.

- Primary Goal: Classify network traffic (Normal, Exploits, DoS, etc.) with high precision.
- The "Why": Uses SHAP (SHapley Additive exPlanations)** to visualize exactly which network features led to a specific threat alert.

## 🛠️ Tech Stack
- AI/ML: TensorFlow, Keras, Scikit-Learn
- Explainability: SHAP
- Dashboard: Streamlit
- Deployment: Docker, Hugging Face Spaces
- Dataset: UNSW-NB15

## ✨ Key Features
- Real-time Analysis: Input network metadata for instant threat classification.
- Transparent Logic: Interactive Force Plots reveal the feature contributions behind every prediction.
- Portability: Fully containerized using Docker for consistent deployment.

## Setup

Use Python 3.10 or newer (the model was saved with Keras 3.10). `scikit-learn` is pinned to 1.6.1, the version that wrote `scaler.pkl` and `label_encoder.pkl`.

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The model, label encoder, and scaler are normal Git files in the repository root (`sentinel_ai_model.keras`, `sentinel_ai_model.h5`, `label_encoder.pkl`, `scaler.pkl`). A clone already contains them; `git lfs pull` is not required.

## Run

```bash
streamlit run app.py
```

Streamlit serves the dashboard at `http://localhost:8501`. The app loads `sentinel_ai_model.keras`, `label_encoder.pkl`, and `scaler.pkl` from the same directory as `app.py` on startup.

### Docker

```bash
docker build -t sentinel-ai .
docker run --rm -p 7860:7860 sentinel-ai
```

Then open `http://localhost:7860`.

---
*Developed as a demonstration of applying Deep Learning and XAI to modern cybersecurity challenges.*
