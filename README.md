# Assessment AI Model — Backend API

AI-powered sustainability assessment predictor using XGBoost. Predicts farmer answers to assessment questions based on farm profile (crop, country, partner, area, irrigation etc.) — trained on 158,311 historical assessment records.

---

## Project Structure

```
assessment_ai_project/
├── src/
│   ├── api_server.py          # Flask REST API (all endpoints)
│   ├── predictor.py           # XGBoost model loading, prediction, active learning
│   ├── model_trainer.py       # Training pipeline (SMOTE, hyperparameter tuning)
│   ├── data_loader.py         # CSV loading and question metadata extraction
│   ├── feature_engineering.py # Feature encoding and transformation
│   ├── database.py            # SQLite persistence (predictions, feedback, model registry)
│   └── config.py              # All paths, model parameters, thresholds
├── data/
│   ├── mock_test_uncopied_data.csv   # Training data (158K rows)
│   ├── mock_test_data.csv            # Small sample (12 rows) for quick testing
│   ├── predicted_assessment.csv      # Legacy export (kept for reference)
│   └── assessment_ai.db             # SQLite database (auto-created on first run)
├── models/
│   └── xgboost_models.pkl           # Trained model file (312MB, 272 classifiers)
├── config/
│   └── crop_synonyms.json           # Localized crop name overrides (e.g. ARROZ → Rice)
├── main.py                           # Standalone training script (run without API)
├── run_api.sh                        # Shell script to start the server
└── test_middleware.py                # Middleware tests
```

---

## Requirements

- **Python 3.9+**
- **pip** (Python package manager)
- macOS / Linux / Windows (WSL)

---

## Setup — Step by Step

### Step 1 — Clone / open the project

```bash
cd /Users/pravinshelke/Documents/AssessmentModel
```

### Step 2 — Create a virtual environment

```bash
python3 -m venv .venv
```

### Step 3 — Activate the virtual environment

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

You should see `(.venv)` in your terminal prompt.

### Step 4 — Install dependencies

```bash
pip install flask flask-cors pandas xgboost scikit-learn numpy joblib shap imbalanced-learn
```

Or if a `requirements.txt` exists:
```bash
pip install -r requirements.txt
```

### Step 5 — Verify the model file exists

```bash
ls -lh assessment_ai_project/models/xgboost_models.pkl
# Should show: 312MB file
```

If the model file does **not** exist, train it first (see [Training the Model](#training-the-model) below).

---

## Running the API Server

### Option A — Using the shell script (recommended)

```bash
cd assessment_ai_project
source /Users/pravinshelke/Documents/AssessmentModel/.venv/bin/activate
bash run_api.sh
```

### Option B — Direct Python command

```bash
cd assessment_ai_project
source /Users/pravinshelke/Documents/AssessmentModel/.venv/bin/activate
python3 -m src.api_server
```

Server starts on: **http://localhost:5001**

> The server takes ~8–10 seconds to start because it loads the 312MB model into memory.

---

## Verify the Server is Running

Open a browser or run:

```bash
curl http://localhost:5001/api/v1/health
```

Expected response:
```json
{
  "status": "healthy",
  "models_loaded": true,
  "total_models": 272,
  "average_accuracy": 0.8458,
  "database": {
    "connected": true,
    "total_prediction_runs": 0,
    "total_feedback_entries": 0
  }
}
```

---

## Training the Model

Only needed if `models/xgboost_models.pkl` does not exist, or you want to retrain on new data.

```bash
cd assessment_ai_project
source /Users/pravinshelke/Documents/AssessmentModel/.venv/bin/activate
python3 main.py
```

This reads `data/mock_test_uncopied_data.csv` (158K rows) and trains 272 XGBoost classifiers — one per assessment indicator. Training takes **15–30 minutes** depending on machine.

The trained model is saved to `models/xgboost_models.pkl`.

> **What `xgboost_models.pkl` contains:** All 272 trained classifiers, label encoders, feature engineering state, question metadata, and KNN training matrix for similar-farm lookup. It is ~312MB because it stores everything needed for inference in one file.

---

## API Endpoints

### Core

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Server health + DB stats |
| GET | `/api/v1/questions` | List all 307 assessment indicators |
| GET | `/api/v1/models` | List loaded models |
| POST | `/api/v1/predict` | **Main: predict answers for one farm** |
| POST | `/api/v1/predict/batch` | Predict for multiple farms at once |
| POST | `/api/v1/predict/from-plan` | Predict using plan context |

### Prediction — Request Body

`POST /api/v1/predict`

```json
{
  "country": "IN",
  "crop": "Wheat",
  "partner": "PartnerA",
  "subpartner": "",
  "irrigation": true,
  "hired_workers": 5,
  "area": 10,
  "planYear": 2024
}
```

### Prediction — Response

```json
{
  "run_id": "abc123-...",
  "predictions": [
    {
      "indicator": "BH-1",
      "predicted_value": "Yes",
      "confidence": 0.92,
      "ui_status": "auto_fill"
    }
  ],
  "sustainability_score": {
    "score": 74,
    "grade": "B"
  },
  "summary": {
    "total_indicators": 272,
    "high_confidence": 198,
    "requires_review": 74
  }
}
```

### Confidence → UI Behaviour

| Confidence | Status | Mobile App Action |
|------------|--------|-------------------|
| ≥ 70% | `auto_fill` | Pre-fills answer, user can skip |
| 40–70% | `suggest` | Shows suggestion, user reviews |
| < 40% | `manual` | User must answer manually |

### Feedback & Learning

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/feedback` | Submit correction when user changes AI answer |
| POST | `/api/v1/active-learning/apply` | Manually trigger model retraining from feedback |
| POST | `/api/v1/retrain` | Full retrain from CSV |

Feedback body:
```json
{
  "indicator": "BH-1",
  "original_prediction": "No",
  "corrected_value": "Yes",
  "farm_context": { "country": "IN", "crop": "Wheat" }
}
```

### Analytics & Explainability

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/explain/<indicator_code>` | SHAP explanation — why the AI predicted this value |
| POST | `/api/v1/similar-farms` | Find 5 farms most similar to this one |
| GET | `/api/v1/stats/sections` | Prediction stats grouped by assessment section |
| GET | `/api/v1/analytics/partner/<name>` | Per-partner accuracy analytics |
| GET | `/api/v1/metrics/indicators` | Per-indicator model accuracy |

### Model Versioning

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/model-versions` | List all saved model checkpoints |
| POST | `/api/v1/model-versions/load` | Roll back to a previous version |

### Database (SQLite History)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/db/stats` | Total predictions, feedback count |
| GET | `/api/v1/db/predictions` | All prediction run history |
| GET | `/api/v1/db/predictions/<run_id>` | Full detail for one run |
| GET | `/api/v1/db/feedback` | All feedback entries |
| GET | `/api/v1/db/model-registry` | All model version registrations |

---

## Database

SQLite database is auto-created at `data/assessment_ai.db` on first server start.

**4 tables:**

| Table | What it stores |
|-------|----------------|
| `prediction_runs` | One row per API `/predict` call — inputs, sustainability score, timestamp |
| `prediction_items` | One row per indicator result within a run |
| `feedback` | Every correction submitted by a user |
| `model_registry` | Every model version trained/saved |

No setup needed — the database is created and managed automatically.

---

## Configuration

Edit `src/config.py` to change behaviour:

| Setting | Default | Description |
|---------|---------|-------------|
| `MIN_TRAINING_SAMPLES` | 20 | Minimum rows needed to train a model for an indicator |
| `CONFIDENCE_AUTO_FILL_THRESHOLD` | 70.0 | Above this → auto-fill in mobile app |
| `CONFIDENCE_SUGGEST_THRESHOLD` | 40.0 | Above this → suggest in mobile app |
| `ENABLE_HYPERPARAMETER_TUNING` | False | Set True for GridSearchCV (slower, more accurate) |
| `MODEL_PARAMS` | see config | XGBoost hyperparameters |

---

## Crop Synonyms

Edit `config/crop_synonyms.json` to map localized crop names to standard names:

```json
{
  "ARROZ": "Rice",
  "AGUACATE": "Avocado",
  "Blé": "Wheat"
}
```

This is only for human-added overrides. Do not add entries where key = value.

---

## Mobile App Integration

The React Native mobile app connects to this server at `http://localhost:5001` (development) or your production URL.

**Flow:**
1. User fills in farm profile (crop, country, partner, area, irrigation)
2. App calls `POST /api/v1/predict`
3. AI predictions pre-fill assessment answers (shown with AI badge)
4. User reviews and corrects if needed
5. Corrections are sent via `POST /api/v1/feedback`
6. Model improves over time via active learning

---

## Troubleshooting

**Server won't start:**
- Make sure the virtual environment is activated: `source .venv/bin/activate`
- Make sure you run from inside `assessment_ai_project/`: `cd assessment_ai_project && python3 -m src.api_server`

**Port already in use:**
```bash
lsof -ti:5001 | xargs kill -9
```

**Model file missing:**
```bash
python3 main.py   # retrain from CSV
```

**Module not found errors:**
```bash
pip install flask flask-cors pandas xgboost scikit-learn numpy joblib shap imbalanced-learn
```
