# Prompt Police: AI Safety Guardrail 🛡️

A real-time binary classifier that catches adversarial (jailbreak) prompts using a dual-stage cascade architecture.

## Overview
This project protects LLMs from malicious prompts hiding in plain sight (roleplay tricks, Base64 encoding, multi-turn manipulation).
It takes a raw user prompt and outputs `SAFE` or `ADVERSARIAL` with a robust confidence score.

The architecture combines rule-based feature extraction (TF-IDF + Regex features) with semantic embeddings (`sentence-transformers`), fed into an XGBoost ensemble classifier.

## Quick Start Guide

**Prerequisites:** Python 3.8 to 3.11 recommended. The `datasets` library will automatically download required datasets.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Models (Training)
Before running the UI, you need to generate the local models. This script will download the datasets from Hugging Face and train the classifier.
```bash
python src/train.py
```
*(This may take a few minutes as it downloads the datasets and trains the models. The artifacts will be saved in the `models/` folder.)*

### 3. Run the Streamlit UI
Start the interactive application to test the prompt bouncer in real-time.
```bash
streamlit run app.py
```

## How It Works
- **Stage 1 (Zone Routing):** Evaluates prompts as LOW, HIGH, or GREY confidence instantly.
- **Stage 2 (Grey Zone Arbitration):** Deeply analyses ambiguous prompts by weighting rule signals (injection patterns, persona overrides) and evaluating conversation context history to catch multi-turn attacks.
