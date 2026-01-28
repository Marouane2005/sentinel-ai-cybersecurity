# 🛡️ Sentinel-AI: Explainable Intrusion Detection

Sentinel-AI is a real-time Network Intrusion Detection System (NIDS) that identifies and explains cyber threats using Deep Learning.

### 🚀 [Live Dashboard on Hugging Face](PASTE_YOUR_LINK_HERE)

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

---
*Developed as a demonstration of applying Deep Learning and XAI to modern cybersecurity challenges.*
