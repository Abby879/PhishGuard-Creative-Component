# PhishGuard — Phishing URL Detection (Creative Component)

PhishGuard is a Streamlit web application that helps users quickly check whether a URL is **Phishing** or **Legitimate**.  
Users paste a link, and the app predicts the result using a trained machine learning model.

## Key Features
- Paste a URL and get an instant prediction (**Phishing / Legitimate**)
- Simple, non-technical UI built with Streamlit
- Uses a saved model artifact (`joblib`) for fast predictions

## Tech Stack
- Python
- Streamlit
- scikit-learn
- pandas, numpy
- joblib

## Project Structure
Phishguard Creative Component
.
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   ├── phishing_model.joblib
│   └── phishing_meta.json
└── assets/
    └── screenshot1.png
