# View this project here:
https://diabetes-risk-segmentation-and-decision.onrender.com/

---

# Diabetes Risk Segmentation & Decision Support System

A machine learning-powered clinical decision support system that combines **XGBoost classification** with **K-means clustering** to stratify diabetes risk and identify personalized intervention opportunities. This project was developed for BC Analytics to deliver evidence-based, actionable insights for patient management.

---

##  Overview

This system addresses two critical clinical challenges:

1. **Diabetes Risk Classification** - Predicts a patient's diabetes stage across five categories (No Diabetes, Low Risk, Prediabetes, Type 1 Diabetes, Type 2 Diabetes) using clinical biomarkers and demographic data.
2. **Lifestyle Segmentation** - Groups patients into three distinct lifestyle profiles to tailor intervention strategies for physical activity, diet quality, and cardiovascular monitoring.

**Key Capabilities:**
- Multi-class diabetes stage prediction with SHAP-based explainability
- Patient segmentation by lifestyle characteristics  
- Interactive web dashboard for real-time predictions and clinical insights
- Handling of severe class imbalance (Type 2: 60% vs. Type 1: <1%)

---

##  Project Structure

```
.
├── README.md                                 # Documentation
├── requirements.txt                          # Python dependencies (pip install)
├── data/
│   ├── Diabetes_and_LifeStyle_Dataset_.csv  # 3,500+ records, 31 features
│   ├── train.csv                            # 80% stratified training split
│   └── test.csv                             # 20% hold-out test set
├── notebooks/                                # Jupyter development notebooks
│   ├── eda_discovery.ipynb                  # Phase 1: Data exploration & profiling
│   ├── k-means-modeling.ipynb               # Phase 2: Clustering model development
│   ├── modeling.ipynb                       # Phase 3: Classification & model comparison
│   └── web_application.ipynb                # Phase 4: Dashboard prototyping
├── src/                                      # Production code
│   ├── prepare_data.py                      # Data loading, outlier clipping, train/test split
│   ├── preprocess_data.py                   # Feature scaling and label encoding
│   ├── train_models.py                      # XGBoost training with class weights (artifact export)
│   ├── web_app.py                           # Dash/Flask application with real-time predictions
│   └── assets/
│       └── custom.css                       # UI styling (dark theme, blue accents)
├── artifacts/                                # Trained models & evaluation artifacts
│   ├── model_1.pkl                          # XGBoost classifier
│   ├── model_dt.pkl, model_rf.pkl           # Baseline models (Decision Tree, Random Forest)
│   ├── kmeans.pkl                           # K-Means clusterer (3 segments)
│   ├── label_encoder.pkl                    # Target variable decoder
│   ├── model_features.pkl                   # Feature column order (critical for prediction)
│   ├── lifestyle_features.pkl               # Features input to K-Means
│   ├── scaler.pkl                           # StandardScaler for lifestyle features
│   ├── feature_importance.csv               # SHAP mean |values| for top 30+ features
│   ├── predictions.csv                      # Hold-out test predictions for validation
│   ├── model_comparison.png                 # Accuracy: Decision Tree vs Random Forest vs XGBoost
│   ├── xgboost_confusion_matrix.png         # Actual vs Predicted for all five stages
│   ├── class_imbalance_chart.png            # Class distribution visualization
│   └── shap_explainer.pkl                   # SHAP TreeExplainer for feature importance
└── scripts/                                  # Utility scripts
```

---

##  Data Overview

**Dataset**: `Diabetes_and_LifeStyle_Dataset_.csv`
- **Records**: 3,500+ patients
- **Features**: 31 (demographics, clinical biomarkers, lifestyle metrics)
- **Target**: `diabetes_stage` (5 classes)

### Key Columns
| Category | Examples |
|----------|----------|
| **Demographics** | Age, gender, ethnicity, education, income, employment, smoking status |
| **Clinical Biomarkers** | Fasting glucose, HbA1c, total cholesterol, triglycerides, systolic/diastolic BP, heart rate |
| **Lifestyle** | Physical activity (min/week), diet score, sleep hours/day, screen time, alcohol consumption |
| **History** | Family history, hypertension, cardiovascular disease |

### Class Distribution (Training Set)
- **Type 2 Diabetes**: ~60% (majority class - well-controlled patients)
- **Prediabetes**: ~32% (intervention opportunity)
- **No Diabetes**: ~8% (preventive/healthy population)
- **Type 1 Diabetes**: <1% (severely underrepresented)
- **Gestational**: <1% (severely underrepresented)

⚠️ **Severe class imbalance addressed via `class_weight='balanced'` in XGBoost training**

---

##  Installation & Setup

### Prerequisites
- Python 3.8+
- Virtual environment (venv, conda, or pipenv)

### Quick Start

1. **Clone and navigate to project**
   ```bash
   git clone <repository-url>
   cd MLG382-Diabetes-Risk-Segmentation-And-Decision-Support-System
   ```

2. **Create and activate virtual environment**
   ```bash
   # Windows
   python -m venv .venv
   .\.venv\Scripts\activate

   # macOS/Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
   
   **Requirements include**: dash, plotly, scikit-learn, xgboost, pandas, joblib, gunicorn

---

## 💻 Usage

### Run the Interactive Dashboard

```bash
python src/web_app.py
```

 Open browser to: **`http://localhost:8050`**

**Dashboard Features:**
- **Model Insights Pane**: View feature importance (SHAP), confusion matrix, class distribution charts
- **Patient Input Form**: Enter demographics and lifestyle metrics; form auto-populates with dataset medians
- **Prediction Results**: Get instant diabetes stage classification + lifestyle segment assignment
- **Clinical Guidance**: Stage-specific and segment-specific recommendations
- **Driver Analysis**: See top 5 global feature drivers (SHAP) and segment-specific characteristics

---

##  Model Development Pipeline

### Phase 1: Exploratory Data Analysis (`eda_discovery.ipynb`)
- ✅ Loaded 3,500+ records; confirmed no missing values
- ✅ Identified severe class imbalance (Type 2: 60%, Type 1: <1%)
- ✅ Correlation analysis: HbA1c, fasting glucose, postprandial glucose strongly predict diabetes stage
- ✅ Confirmed data leakage risk: `diagnosed_diabetes` and `diabetes_risk_score` dropped
- ✅ Lifestyle features (physical activity, diet) show weaker but meaningful correlations

### Phase 2: Clustering (`k-means-modeling.ipynb`)
- K-Means initialized with `k=3` lifestyle segments
- Standardized features: physical activity, diet score, sleep, screen time, alcohol
- Generated segment center profiles and silhouette validation
- Exported `kmeans.pkl` and `scaler.pkl` for web app

### Phase 3: Classification (`modeling.ipynb`)
- **Baseline**: Decision Tree (`max_depth=10`)
- **Ensemble**: Random Forest (100 trees, `class_weight='balanced'`)
- **Final Model**: XGBoost (200 estimators, `max_depth=6`, `learning_rate=0.1`)
  - Weighted samples to penalize minority class misclassification
  - Achieved best hold-out accuracy and balanced precision/recall across stages
- Exported artifacts: `model_1.pkl`, `label_encoder.pkl`, `model_features.pkl`
- Generated SHAP feature importance: `feature_importance.csv`
- Predictions saved: `predictions.csv`

### Phase 4: Web Application (`web_application.ipynb`)
- Dash-based interactive dashboard
- Real-time prediction callback with form validation
- Plotly visualizations (bar charts, heatmaps, cluster centers)
- Dark theme CSS with blue accent colors

---

##  Model Details

### Classification: XGBoost

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Algorithm** | Gradient Boosting (XGBoost) | Handles non-linearity, feature interactions, imbalanced data |
| **n_estimators** | 200 | Sufficient boosting rounds without overfitting |
| **max_depth** | 6 | Moderate tree depth; prevents memorization |
| **learning_rate** | 0.1 | Conservative updates; stable convergence |
| **eval_metric** | mlogloss | Multi-class cross-entropy loss |
| **class_weight** | balanced | Penalizes minority class errors; addresses ~1% Type 1 prevalence |

**Predictions:** Returns encoded stage + confidence scores

**Explainability:** SHAP TreeExplainer computes mean |SHAP values| per feature to rank influence on predictions

### Segmentation: K-Means

| Parameter | Value |
|-----------|-------|
| **Algorithm** | K-Means clustering |
| **k** | 3 lifestyle segments |
| **Features** | physical_activity, diet_score, sleep_hours, screen_time, alcohol_consumption (standardized) |
| **Output** | Segment ID (0, 1, 2) + cluster center characteristics |

---

##  Clinical Recommendations

### Diabetes Stage Guidance

| Stage | Recommendation |
|-------|-----------------|
| **No Diabetes** | Continue preventive lifestyle habits; routine screening annually |
| **Low Risk** | Maintain current healthy habits; annual screening sufficient |
| **Prediabetes** | Increase physical activity, improve diet quality; re-test glucose in 3–6 months |
| **Type 2 Diabetes** | Structured lifestyle plan + medication adherence; monitor HbA1c every 3 months |
| **Type 1 Diabetes** | Coordinate insulin management with clinician; intensive glucose monitoring |

### Lifestyle Segment Guidance

| Segment | Focus Area | Intervention |
|---------|-----------|--------------|
| **0** | Activity & Sleep | Prioritize consistent physical activity (goal: 150+ min/week) and 7–8 hrs sleep nightly |
| **1** | Diet & Screen Time | Improve diet quality (target: score 7+); reduce prolonged sitting and screen exposure |
| **2** | Cardiometabolic Risk | Close medical follow-up; monitor BP, cholesterol, triglycerides; emphasize preventive medication adherence |

---

##  Deployment to Render

### Website url
```
    `https://diabetes-risk-segmentation-and-decision.onrender.com/`
```

##  Notebook Guide

| Notebook | Purpose | Outputs |
|----------|---------|---------|
| **eda_discovery.ipynb** | Data exploration, profiling, leakage detection | Correlation heatmap, distribution plots, class balance charts |
| **k-means-modeling.ipynb** | Cluster development, segment profiling | kmeans.pkl, scaler.pkl, segment visualizations |
| **modeling.ipynb** | Model comparison (DT/RF/XGBoost), SHAP analysis | model_1.pkl, feature_importance.csv, confusion matrix PNG |
| **web_application.ipynb** | Dashboard interface design & prototyping | Layout mockups, callback patterns, styling examples |

---

## 🛠 Troubleshooting

### Issue: Model predictions fail / "NotFittedError"
- **Check**: All artifact files present in `/artifacts`
- **Fix**: Re-run `python src/train_models.py` to regenerate models

### Issue: Form input validation error
- **Check**: Dataset at `data/Diabetes_and_LifeStyle_Dataset_.csv` is accessible
- **Fix**: Verify file path and encoding (UTF-8)

### Issue: Render deployment timeout
- **Check**: Gunicorn worker timeout setting
- **Fix**: Increase `--timeout 120` in start command if models load slowly

---

##  Performance Summary

- **Classification Accuracy**: ~82–85% on hold-out test set (average across 5 classes)
- **Class-Specific F1 Scores**: Reported in `modeling.ipynb` classification report
- **Confusion Matrix**: Available in `artifacts/xgboost_confusion_matrix.png`
- **SHAP Top Features**: HbA1c, fasting glucose, postprandial glucose dominate (expected clinical biomarkers)

---



##  Technical Architecture

- **Backend**: Flask (via Dash callbacks)
- **Frontend**: Dash/Plotly with React components
- **ML Stack**: scikit-learn, XGBoost, numpy
- **UI**: Custom dark-theme CSS; Plotly dark template
- **Model Persistence**: joblib pickles (fast, Python-native)
- **Server**: Gunicorn for production WSGI serving

---
