# ⚖️ AI Bias Analysis API

A fairness-aware machine learning API built with Flask, trained on the **COMPAS dataset** to detect and mitigate racial bias in criminal risk assessment predictions.

## 🎯 Project Overview

This project demonstrates the complete lifecycle of responsible AI:
- **Bias detection** using fairness metrics (SPD, DPR, EOD)
- **Bias mitigation** using Fairlearn's pre/in/post-processing techniques
- **Explainability** via SHAP analysis
- **Continuous monitoring** for fairness drift

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| **Model** | Fairlearn Mitigated Classifier |
| **Accuracy** | ~0.65 |
| **Statistical Parity Difference (SPD)** | Near 0 (fair) |
| **Sensitive Feature** | `race_binary` (African-American vs Others) |

## 🔌 API Endpoints

### `GET /`
Returns an HTML home page with model info.

### `POST /predict`
Make a prediction on a single sample.

**Request (recommended — named features):**
```json
{
  "age": 30,
  "c_charge_degree": "F",
  "priors_count": 0,
  "race": "African-American",
  "sex": "Male",
  "juv_fel_count": 0,
  "juv_misd_count": 0
}

Alternative — raw feature array:
{
  "features": [30, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}

Response:
{
  "prediction": 0,
  "prediction_label": "Low Risk",
  "probability": [0.72, 0.28],
  "model_accuracy": 0.6531
}

Field reference:

Field	Type	Description
age	integer	Defendant's age in years
c_charge_degree	string	"F" (felony) or "M" (misdemeanor)
priors_count	integer	Number of prior convictions
race	string	"African-American", "Caucasian", "Hispanic", "Other", or "Asian"
sex	string	"Male" or "Female"
juv_fel_count	integer	Number of juvenile felony counts
juv_misd_count	integer	Number of juvenile misdemeanor counts

GET /fairness
Returns fairness metrics for the deployed model.

Response:
{
  "model_name": "...",
  "accuracy": 0.65,
  "spd": 0.03,
  "sensitive_feature": "race_binary",
  "note": "SPD closer to 0 means fairer model"
}

GET /health
Health check endpoint. Returns {"status": "healthy"}.

🛠️ Tech Stack
Python 3.10

Flask — Web framework

scikit-learn — ML models

Fairlearn — Bias mitigation

SHAP — Model explainability

XGBoost — Gradient boosting

Gunicorn — WSGI server

📦 Dataset
COMPAS Recidivism Dataset — 7,214 defendants with criminal history features.

🚀 Deployment
Deployed on Render (free tier).

📁 Project Structure
bias-analysis-api/
├── app.py                  # Flask application
├── requirements.txt        # Python dependencies
├── bias_model.pkl          # Trained fairness-mitigated model
├── preprocessor.pkl        # Data preprocessing pipeline
├── model_info.json         # Model metadata
└── README.md

🔬 Methodology
Data Loading — COMPAS dataset (7,214 samples, 10 features)

Preprocessing — StandardScaler for numerical, OneHotEncoder for categorical

Baseline Models — Logistic Regression, Random Forest, XGBoost, Neural Network, SVM

Bias Detection — Fairlearn metrics across sensitive groups

Mitigation — Reweighting, Demographic Parity, Equalized Odds, Threshold Optimizer

Explainability — SHAP analysis on top features

Monitoring — Drift detection over simulated time windows

📖 Key Findings
Baseline models showed significant bias: SPD ~0.33, EOD ~0.38

Mitigation reduced SPD to ~0.03 (near-fair) with minimal accuracy loss

Top predictive features: priors_count, age, race

Race appearing as a top feature confirms historical bias in the data

👤 Author
Isha Bhagat — Major Project, SmartED Innovations

📄 License
MIT License



