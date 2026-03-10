from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import os
import json
import uuid
import datetime
import threading
import time
import pandas as pd
from .predictor import AssessmentPredictor
from .config import Config
from .database import AssessmentDB

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# ── Initialize predictor ──────────────────────────────────────────
try:
    predictor = AssessmentPredictor()
    try:
        predictor.load_checkpoint()
        if not predictor.models:
            raise RuntimeError("Checkpoint contained no trained models")
        logger.info("Loaded pre-trained models.")
    except Exception as load_err:
        logger.info(f"Checkpoint not usable ({load_err}), training models on startup...")
        predictor.run_training_pipeline()
except Exception as e:
    logger.error(f"Failed to initialize predictor: {e}")
    predictor = None

# ── Initialize database ───────────────────────────────────────────
try:
    db = AssessmentDB(Config.DB_FILE)
    logger.info(f"Database initialized: {Config.DB_FILE}")
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")
    db = None

# ── Active Learning Background Thread ────────────────────────────
_active_learning_lock = threading.Lock()

def _active_learning_loop(interval_seconds=3600):
    """Runs every `interval_seconds` (default 1 hour). Retrains models with enough feedback."""
    while True:
        time.sleep(interval_seconds)
        if predictor is None:
            continue
        with _active_learning_lock:
            try:
                result = predictor.apply_feedback_to_training(
                    min_corrections=10, db=db
                )
                if result.get('retrained'):
                    logger.info(f"Active learning retrained: {result['retrained']}")
            except Exception as e:
                logger.error(f"Active learning loop error: {e}")

_al_thread = threading.Thread(target=_active_learning_loop, daemon=True)
_al_thread.start()
logger.info("Active learning background thread started (runs every 1 hour)")


# ── Health ────────────────────────────────────────────────────────
@app.route('/api/v1/health', methods=['GET'])
def health_check():
    stats = predictor.get_model_stats() if predictor else {}
    db_stats = db.get_stats() if db else {}
    return jsonify({
        "status": "healthy",
        "service": "Assessment AI Model",
        "models_loaded": predictor is not None and len(predictor.models) > 0,
        "total_models": stats.get('total_models', 0),
        "average_accuracy": stats.get('average_accuracy', 0),
        "similar_farms_available": predictor is not None and predictor.X_encoded is not None,
        "database": {
            "connected": db is not None,
            "path": Config.DB_FILE,
            **db_stats,
        },
    })


# ── Models ────────────────────────────────────────────────────────
@app.route('/api/v1/models', methods=['GET'])
def model_stats():
    if not predictor:
        return jsonify({"success": False, "error": "Model not initialized"}), 500
    stats = predictor.get_model_stats()
    return jsonify({"success": True, **stats})


# ── Questions ─────────────────────────────────────────────────────
@app.route('/api/v1/questions', methods=['GET'])
def get_questions():
    trained_indicators = set(predictor.models.keys()) if predictor else set()
    questions = []
    for code, meta in Config.QUESTION_META.items():
        has_model = code in trained_indicators
        questions.append({
            "indicatorCode": code,
            "labelName": meta.get("labelName", ""),
            "type": meta.get("type"),
            "section": meta.get("section", "General"),
            "aiAvailable": has_model or meta.get("type") not in ("text", "number"),
            "manualEntry": not has_model and meta.get("type") in ("text", "number"),
            "options": meta.get("options", [])
        })
    return jsonify({"success": True, "total_questions": len(questions), "questions": questions})


# ── Predict ───────────────────────────────────────────────────────
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    if not predictor:
        return jsonify({"success": False, "error": "Model not initialized"}), 500
    try:
        data = request.json
        if not data:
            return jsonify({"success": False, "error": "No JSON data provided"}), 400

        input_data = {
            'country_code': data.get('country'),
            'crop_name':    data.get('crop'),
            'partner':      data.get('partner'),
            'subpartner':   data.get('subpartner'),
            'irrigation':   data.get('irrigation'),
            'hired_workers':data.get('hired_workers'),
            'area':         data.get('area'),
            'planYear':     data.get('planYear', data.get('plan_year')),
        }

        predictions = predictor.predict(input_data)

        confidences = [p['confidence'] for p in predictions.values()]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        high_conf = len([c for c in confidences if c >= Config.CONFIDENCE_THRESHOLD])

        auto_filled_questions, needs_review_questions = [], []
        for indicator, res in predictions.items():
            meta = Config.QUESTION_META.get(indicator, {})
            item = {"indicator": indicator, "section": meta.get("section", "General"), "prediction": res}
            if res.get("ui_status") == "green":
                auto_filled_questions.append(item)
            else:
                needs_review_questions.append(item)

        predictions_by_label = {}
        for indicator, res in predictions.items():
            label = Config.QUESTION_META.get(indicator, {}).get("labelName", "").strip()
            if label:
                predictions_by_label[label] = {**res, "indicatorCode": indicator}

        # Sustainability score (always included)
        score = predictor.compute_sustainability_score(predictions)

        # Optional year-over-year comparison
        yoy = None
        prev_year = data.get('previousPlanYear')
        if prev_year:
            try:
                yoy = predictor.compare_year_over_year(predictions, prev_year, input_data)
            except Exception as e:
                logger.warning(f"YoY comparison failed: {e}")
                yoy = {"error": str(e)}

        summary = {
            "total_indicators": len(predictions),
            "auto_filled_count": len(auto_filled_questions),
            "needs_review_count": len(needs_review_questions),
            "average_confidence": round(avg_conf, 1),
            "saved_time_estimate": f"{len(auto_filled_questions) * 0.5} minutes",
        }

        # Persist to database (best-effort — never block the API response)
        run_id = str(uuid.uuid4())
        if db:
            try:
                db.save_prediction_run(
                    run_id=run_id,
                    input_data=input_data,
                    predictions=predictions,
                    summary=summary,
                    sustainability_score=score,
                    question_meta=Config.QUESTION_META,
                )
            except Exception as db_err:
                logger.warning(f"DB save failed (non-fatal): {db_err}")

        return jsonify({
            "success": True,
            "run_id": run_id,
            "statistics": {
                "total_indicators": len(predictions),
                "high_confidence": high_conf,
                "average_confidence": avg_conf,
            },
            "summary": summary,
            "sustainability_score": score,
            "year_over_year": yoy,
            "groups": {"auto_filled": auto_filled_questions, "needs_review": needs_review_questions},
            "predictions": predictions,
            "predictions_by_label": predictions_by_label,
            "raw_predictions": predictions,
            "metadata": data,
        })
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Batch Predict ─────────────────────────────────────────────────
@app.route('/api/v1/predict/batch', methods=['POST'])
def predict_batch():
    if not predictor:
        return jsonify({"success": False, "error": "Model not initialized"}), 500
    try:
        body = request.json
        records = body.get('records', [])
        if not records or not isinstance(records, list):
            return jsonify({"success": False, "error": "Provide a 'records' array"}), 400
        if len(records) > 100:
            return jsonify({"success": False, "error": "Max 100 records per batch"}), 400
        results = []
        for rec in records:
            input_data = {
                'country_code': rec.get('country'), 'crop_name': rec.get('crop'),
                'partner': rec.get('partner'), 'subpartner': rec.get('subpartner'),
                'irrigation': rec.get('irrigation'), 'hired_workers': rec.get('hired_workers'),
                'area': rec.get('area'), 'planYear': rec.get('planYear', rec.get('plan_year')),
            }
            preds = predictor.predict(input_data)
            confs = [p['confidence'] for p in preds.values()]
            score = predictor.compute_sustainability_score(preds)
            results.append({
                "input": rec, "predictions": preds, "sustainability_score": score,
                "average_confidence": round(sum(confs) / len(confs), 2) if confs else 0,
            })
        return jsonify({"success": True, "count": len(results), "results": results})
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Section Stats ─────────────────────────────────────────────────
@app.route('/api/v1/stats/sections', methods=['GET'])
def section_stats():
    if not predictor:
        return jsonify({"success": False, "error": "Model not initialized"}), 500
    section_data = {}
    for indicator, packet in predictor.models.items():
        section = Config.QUESTION_META.get(indicator, {}).get('section', 'General')
        acc = packet.get('accuracy', 0)
        section_data.setdefault(section, {'indicators': 0, 'accuracy_sum': 0.0})
        section_data[section]['indicators'] += 1
        section_data[section]['accuracy_sum'] += acc
    sections = [{"section": s, "indicator_count": d['indicators'],
                  "average_accuracy": round(d['accuracy_sum'] / d['indicators'] * 100, 1)}
                 for s, d in sorted(section_data.items())]
    return jsonify({"success": True, "sections": sections})


# ── SHAP Explain ──────────────────────────────────────────────────
@app.route('/api/v1/explain/<string:indicator_code>', methods=['POST'])
def explain_indicator(indicator_code):
    """
    Returns SHAP feature contributions explaining why the AI predicted a particular value.
    Body: same as /predict (country, crop, partner, ...)
    """
    if not predictor:
        return jsonify({"success": False, "error": "Model not initialized"}), 500
    try:
        data = request.json or {}
        input_data = {
            'country_code': data.get('country'), 'crop_name': data.get('crop'),
            'partner': data.get('partner'), 'subpartner': data.get('subpartner'),
            'irrigation': data.get('irrigation'), 'hired_workers': data.get('hired_workers'),
            'area': data.get('area'), 'planYear': data.get('planYear', data.get('plan_year')),
        }
        result = predictor.explain_prediction(indicator_code, input_data)
        if 'error' in result:
            return jsonify({"success": False, "error": result['error']}), 404
        return jsonify({"success": True, **result})
    except Exception as e:
        logger.error(f"Explain error for {indicator_code}: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Similar Farms ─────────────────────────────────────────────────
@app.route('/api/v1/similar-farms', methods=['POST'])
def similar_farms():
    """
    Returns top-N historically similar farms from training data.
    Body: same as /predict. Optional: { "n": 5 }
    """
    if not predictor:
        return jsonify({"success": False, "error": "Model not initialized"}), 500
    try:
        data = request.json or {}
        input_data = {
            'country_code': data.get('country'), 'crop_name': data.get('crop'),
            'partner': data.get('partner'), 'subpartner': data.get('subpartner'),
            'irrigation': data.get('irrigation'), 'hired_workers': data.get('hired_workers'),
            'area': data.get('area'), 'planYear': data.get('planYear', data.get('plan_year')),
        }
        n = min(int(data.get('n', 5)), 20)
        result = predictor.find_similar_farms(input_data, n=n)
        if 'error' in result:
            return jsonify({"success": False, "error": result['error']}), 503
        return jsonify({"success": True, **result})
    except Exception as e:
        logger.error(f"Similar farms error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Feedback ──────────────────────────────────────────────────────
@app.route('/api/v1/feedback', methods=['POST'])
def collect_feedback():
    try:
        data = request.json
        if not data or not data.get('indicator') or not data.get('actual'):
            return jsonify({"success": False, "error": "Requires 'indicator' and 'actual' fields"}), 400

        # Save to database (primary store)
        row_id = None
        count = 0
        if db:
            row_id = db.save_feedback(data)
            count = db.get_feedback_counts()
            count = next((r['total'] for r in count if r['indicator'] == data.get('indicator')), 1)
        else:
            # Fallback: write to JSONL if DB unavailable
            entry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "indicator": data.get('indicator'), "predicted": data.get('predicted'),
                "actual": data.get('actual'), "confidence": data.get('confidence'),
                "input": data.get('input', {}),
            }
            feedback_file = os.path.join(Config.DATA_DIR, 'feedback.jsonl')
            with open(feedback_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(entry) + '\n')

        logger.info(f"Feedback recorded: {data.get('indicator')} → {data.get('actual')}")
        return jsonify({"success": True, "message": "Feedback saved.", "row_id": row_id, "corrections_for_indicator": count})
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Partner Analytics ─────────────────────────────────────────────
@app.route('/api/v1/analytics/partner/<string:partner_name>', methods=['GET'])
def partner_analytics(partner_name):
    """
    Aggregated analytics for a specific partner:
    - Model accuracy per section
    - Total feedback/corrections submitted
    - Most-corrected indicators
    - Countries and crops observed for this partner
    """
    if not predictor:
        return jsonify({"success": False, "error": "Model not initialized"}), 500
    try:
        # Section accuracy for models (global — not partner-specific at model level)
        section_data = {}
        for indicator, packet in predictor.models.items():
            section = Config.QUESTION_META.get(indicator, {}).get('section', 'General')
            section_data.setdefault(section, {'count': 0, 'acc_sum': 0.0})
            section_data[section]['count'] += 1
            section_data[section]['acc_sum'] += packet.get('accuracy', 0)

        section_summary = {s: round(v['acc_sum'] / v['count'] * 100, 1)
                           for s, v in section_data.items()}

        # Feedback stats for this partner (read from DB, fall back to JSONL)
        partner_feedback = []
        if db:
            all_fb = db.get_all_feedback_as_jsonl_entries()
            partner_feedback = [
                e for e in all_fb
                if str(e.get('input', {}).get('partner', '')).lower() == partner_name.lower()
            ]
        else:
            feedback_file = os.path.join(Config.DATA_DIR, 'feedback.jsonl')
            if os.path.exists(feedback_file):
                with open(feedback_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            e = json.loads(line)
                            inp = e.get('input', {})
                            if str(inp.get('partner', '')).lower() == partner_name.lower():
                                partner_feedback.append(e)
                        except Exception:
                            continue

        # Most corrected indicators
        correction_counts = {}
        for e in partner_feedback:
            ind = e.get('indicator', 'unknown')
            correction_counts[ind] = correction_counts.get(ind, 0) + 1
        most_corrected = sorted(correction_counts.items(), key=lambda x: x[1], reverse=True)[:10]

        # Countries and crops from training data
        countries, crops = set(), set()
        if predictor.df_raw is not None:
            mask = predictor.df_raw.get('partner', pd.Series()).str.lower() == partner_name.lower()
            partner_rows = predictor.df_raw[mask] if hasattr(mask, '__len__') else predictor.df_raw
            countries = set(partner_rows.get('country_code', pd.Series()).dropna().unique().tolist())
            crops = set(partner_rows.get('crop_name', pd.Series()).dropna().unique().tolist())

        return jsonify({
            "success": True,
            "partner": partner_name,
            "total_models": len(predictor.models),
            "section_accuracy": section_summary,
            "feedback": {
                "total_corrections": len(partner_feedback),
                "most_corrected_indicators": [{"indicator": k, "count": v} for k, v in most_corrected],
            },
            "training_data": {
                "countries": sorted(countries),
                "crops": sorted(crops),
            },
        })
    except Exception as e:
        logger.error(f"Partner analytics error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Retrain ───────────────────────────────────────────────────────
@app.route('/api/v1/retrain', methods=['POST'])
def retrain():
    if not predictor:
        return jsonify({"success": False, "error": "Predictor not initialized"}), 500
    try:
        logger.info("Retraining triggered via API...")
        predictor.run_training_pipeline()
        stats = predictor.get_model_stats()
        return jsonify({"success": True, "message": "Retraining complete.",
                        "total_models": stats.get('total_models', 0),
                        "average_accuracy": stats.get('average_accuracy', 0)})
    except Exception as e:
        logger.error(f"Retraining error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Active Learning — manual trigger ─────────────────────────────
@app.route('/api/v1/active-learning/apply', methods=['POST'])
def apply_active_learning():
    """
    Manually trigger active learning: retrain models that have ≥ min_corrections feedback.
    Body: { "min_corrections": 10 }
    """
    if not predictor:
        return jsonify({"success": False, "error": "Predictor not initialized"}), 500
    try:
        with _active_learning_lock:
            min_c = int((request.json or {}).get('min_corrections', 10))
            result = predictor.apply_feedback_to_training(min_corrections=min_c, db=db)
        return jsonify({"success": True, **result})
    except Exception as e:
        logger.error(f"Active learning apply error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Single Indicator Predict ──────────────────────────────────────
@app.route('/api/v1/indicators/<string:indicator_code>', methods=['POST'])
def predict_single_indicator(indicator_code):
    if not predictor:
        return jsonify({"success": False, "error": "Model not initialized"}), 500
    try:
        data = request.json or {}
        input_data = {
            'country_code': data.get('country'), 'crop_name': data.get('crop'),
            'partner': data.get('partner'), 'subpartner': data.get('subpartner'),
            'irrigation': data.get('irrigation'), 'hired_workers': data.get('hired_workers'),
            'area': data.get('area'), 'planYear': data.get('planYear', data.get('plan_year')),
        }
        all_predictions = predictor.predict(input_data)
        result = all_predictions.get(indicator_code)
        if result is None:
            return jsonify({"success": False, "error": f"No model for indicator: {indicator_code}"}), 404
        meta = Config.QUESTION_META.get(indicator_code, {})
        return jsonify({"success": True, "indicator": indicator_code,
                        "section": meta.get('section', 'General'),
                        "label": meta.get('labelName', ''), "prediction": result})
    except Exception as e:
        logger.error(f"Single indicator prediction error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Metrics Dashboard ─────────────────────────────────────────────
@app.route('/api/v1/metrics/indicators', methods=['GET'])
def metrics_dashboard():
    """
    Per-indicator precision, recall, F1, accuracy, confusion matrix.
    Optional query param: ?section=Crop+Management
    """
    if not predictor:
        return jsonify({"success": False, "error": "Model not initialized"}), 500

    section_filter = request.args.get('section', '').strip().lower()
    stats = predictor.get_model_stats()
    rows = []
    for indicator, m in stats.get('per_indicator', {}).items():
        section = Config.QUESTION_META.get(indicator, {}).get('section', 'General')
        if section_filter and section.lower() != section_filter:
            continue
        rows.append({
            "indicator": indicator,
            "section": section,
            "classes": m.get('classes', []),
            "num_classes": m.get('num_classes'),
            "training_samples": m.get('training_samples'),
            "accuracy":  round((m.get('accuracy')  or 0) * 100, 1),
            "f1_score":  round((m.get('f1_score')  or 0) * 100, 1),
            "precision": round((m.get('precision') or 0) * 100, 1),
            "recall":    round((m.get('recall')    or 0) * 100, 1),
            "confusion_matrix": m.get('confusion_matrix'),
        })

    rows.sort(key=lambda r: r['accuracy'], reverse=True)
    return jsonify({
        "success": True,
        "total_indicators": len(rows),
        "summary": {
            "average_accuracy":  round(sum(r['accuracy']  for r in rows) / len(rows), 1) if rows else 0,
            "average_precision": round(sum(r['precision'] for r in rows) / len(rows), 1) if rows else 0,
            "average_recall":    round(sum(r['recall']    for r in rows) / len(rows), 1) if rows else 0,
            "average_f1":        round(sum(r['f1_score']  for r in rows) / len(rows), 1) if rows else 0,
        },
        "indicators": rows,
    })


# ── Model Version Management ──────────────────────────────────────
@app.route('/api/v1/model-versions', methods=['GET'])
def list_model_versions():
    """Lists all versioned checkpoints saved during past retraining runs."""
    versions = Config.list_model_versions()
    return jsonify({"success": True, "versions": versions, "count": len(versions)})


@app.route('/api/v1/model-versions/load', methods=['POST'])
def load_model_version():
    """
    Roll back to a specific saved version.
    Body: { "version": "v_20260310_143000" }
    """
    if not predictor:
        return jsonify({"success": False, "error": "Predictor not initialized"}), 500
    version = (request.json or {}).get('version', '').strip()
    if not version:
        return jsonify({"success": False, "error": "Provide a 'version' field"}), 400
    try:
        predictor.load_checkpoint_version(version)
        stats = predictor.get_model_stats()
        return jsonify({
            "success": True,
            "loaded_version": version,
            "total_models": stats['total_models'],
            "average_accuracy": stats['average_accuracy'],
        })
    except FileNotFoundError as e:
        return jsonify({"success": False, "error": str(e)}), 404
    except Exception as e:
        logger.error(f"Version load error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


# ── Plan Context — how to integrate with a real Plan API ──────────
@app.route('/api/v1/plan-context', methods=['GET'])
def plan_context():
    """
    Explains what input fields the AI model needs and where the frontend/Plan API
    should provide them.  When you connect a real Plan API, replace the placeholder
    in /api/v1/predict/from-plan below with a real HTTP call.
    """
    return jsonify({
        "success": True,
        "description": (
            "The AI prediction model requires plan-level context fields. "
            "Currently these must be supplied by the frontend when opening a new assessment. "
            "Connect a real Plan API by implementing the /api/v1/predict/from-plan endpoint."
        ),
        "required_fields": [
            {"field": "country",       "maps_to": "country_code",   "example": "IN",        "source": "Plan API / user selection"},
            {"field": "crop",          "maps_to": "crop_name",      "example": "Wheat",     "source": "Plan API / crop selection"},
            {"field": "partner",       "maps_to": "partner",        "example": "HRK-TEST",  "source": "Plan API"},
            {"field": "subpartner",    "maps_to": "subpartner",     "example": "SUB-01",    "source": "Plan API (optional — 97% null in training data)"},
            {"field": "irrigation",    "maps_to": "irrigation",     "example": True,        "source": "Plan API / farm details"},
            {"field": "hired_workers", "maps_to": "hired_workers",  "example": 5,           "source": "Plan API / farm details"},
            {"field": "area",          "maps_to": "area",           "example": 20,          "source": "Plan API / farm details"},
            {"field": "planYear",      "maps_to": "planYear",       "example": 2024,        "source": "Plan API (plan creation year)"},
        ],
        "how_to_call": {
            "endpoint": "POST /api/v1/predict",
            "example_body": {
                "country": "IN", "crop": "Wheat", "partner": "HRK-TEST",
                "subpartner": None, "irrigation": True,
                "hired_workers": 5, "area": 20, "planYear": 2024,
            },
        },
        "future_features_to_add_to_csv": [
            "season (Kharif/Rabi/Spring/Summer)",
            "soil_type (Clay/Loam/Sandy/Silt)",
            "farm_age (years, numeric)",
            "certification_status (Organic/Conventional/Transitioning)",
        ],
        "note": (
            "Future features are commented out in Config.FEATURE_COLUMNS. "
            "Add those columns to your training CSV first, then uncomment them."
        ),
    })


@app.route('/api/v1/predict/from-plan', methods=['POST'])
def predict_from_plan():
    """
    Fetches plan details from your external Plan API and runs AI predictions in one call.
    
    Body: { "planId": "PLAN-123" }
    Env vars required:  PLAN_API_BASE_URL   e.g. https://your-plan-api.example.com
                        PLAN_API_TOKEN      Bearer token (optional if public)

    Field mapping (Plan API response → AI model input):
        countryCode   → country_code
        cropName      → crop_name
        partnerCode   → partner
        subPartnerCode→ subpartner
        irrigationType→ irrigation  (True when value != 'rainfed')
        hiredWorkers  → hired_workers
        farmArea      → area
        planYear      → planYear

    Adjust the field names below to match your actual Plan API response shape.
    """
    if not predictor:
        return jsonify({"success": False, "error": "Model not initialized"}), 500

    body = request.json or {}
    plan_id = body.get('planId', '').strip()
    if not plan_id:
        return jsonify({"success": False, "error": "Provide a 'planId' field"}), 400

    plan_api_base = os.environ.get('PLAN_API_BASE_URL', '').rstrip('/')
    if not plan_api_base:
        return jsonify({
            "success": False,
            "error": (
                "PLAN_API_BASE_URL environment variable is not set. "
                "Set it to your Plan API base URL (e.g. https://your-plan-api.example.com) "
                "and optionally set PLAN_API_TOKEN."
            ),
            "setup_guide": "/api/v1/plan-context",
        }), 501

    try:
        import urllib.request as _urlreq
        import urllib.error as _urlerr

        token = os.environ.get('PLAN_API_TOKEN', '')
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['Authorization'] = f'Bearer {token}'

        req = _urlreq.Request(
            f"{plan_api_base}/plans/{plan_id}",
            headers=headers,
            method='GET',
        )
        with _urlreq.urlopen(req, timeout=10) as resp:
            plan = json.loads(resp.read().decode('utf-8'))

    except Exception as fetch_err:
        logger.error(f"Plan API fetch failed for planId={plan_id}: {fetch_err}")
        return jsonify({
            "success": False,
            "error": f"Could not fetch plan from Plan API: {fetch_err}",
            "planId": plan_id,
        }), 502

    # Map Plan API fields → AI model fields
    # Adjust key names here to match your actual Plan API response
    irrigation_raw = plan.get('irrigationType', '')
    input_data = {
        'country_code': plan.get('countryCode') or plan.get('country_code') or plan.get('country'),
        'crop_name':    plan.get('cropName')    or plan.get('crop_name')    or plan.get('crop'),
        'partner':      plan.get('partnerCode') or plan.get('partner'),
        'subpartner':   plan.get('subPartnerCode') or plan.get('subpartner'),
        'irrigation':   irrigation_raw not in ('rainfed', 'none', '', None, False),
        'hired_workers':plan.get('hiredWorkers') or plan.get('hired_workers'),
        'area':         plan.get('farmArea')     or plan.get('area'),
        'planYear':     plan.get('planYear')     or plan.get('plan_year'),
    }

    try:
        predictions = predictor.predict(input_data)
        confidences = [p['confidence'] for p in predictions.values()]
        avg_conf = sum(confidences) / len(confidences) if confidences else 0
        score = predictor.compute_sustainability_score(predictions)

        auto_filled, needs_review = [], []
        for indicator, res in predictions.items():
            meta = Config.QUESTION_META.get(indicator, {})
            item = {"indicator": indicator, "section": meta.get("section", "General"), "prediction": res}
            if res.get("ui_status") == "green":
                auto_filled.append(item)
            else:
                needs_review.append(item)

        return jsonify({
            "success": True,
            "planId": plan_id,
            "plan_context": input_data,
            "summary": {
                "total_indicators": len(predictions),
                "auto_filled_count": len(auto_filled),
                "needs_review_count": len(needs_review),
                "average_confidence": round(avg_conf, 1),
                "saved_time_estimate": f"{len(auto_filled) * 0.5} minutes",
            },
            "sustainability_score": score,
            "groups": {"auto_filled": auto_filled, "needs_review": needs_review},
            "predictions": predictions,
        })
    except Exception as e:
        logger.error(f"Prediction error in from-plan: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)


# ── Database / History Endpoints ──────────────────────────────────

@app.route('/api/v1/db/stats', methods=['GET'])
def db_stats():
    """Quick summary of what's stored in the database."""
    if not db:
        return jsonify({"success": False, "error": "Database not available"}), 503
    return jsonify({"success": True, **db.get_stats()})


@app.route('/api/v1/db/predictions', methods=['GET'])
def prediction_history():
    """
    Return recent prediction runs stored in the database.
    Query params: limit (default 50), crop, partner
    """
    if not db:
        return jsonify({"success": False, "error": "Database not available"}), 503
    limit = int(request.args.get('limit', 50))
    crop = request.args.get('crop')
    partner = request.args.get('partner')
    rows = db.get_prediction_history(limit=limit, crop=crop, partner=partner)
    return jsonify({"success": True, "count": len(rows), "predictions": rows})


@app.route('/api/v1/db/predictions/<string:run_id>', methods=['GET'])
def prediction_run_detail(run_id):
    """Return full indicator-level detail for a specific prediction run."""
    if not db:
        return jsonify({"success": False, "error": "Database not available"}), 503
    items = db.get_prediction_items(run_id)
    if not items:
        return jsonify({"success": False, "error": "Run not found"}), 404
    return jsonify({"success": True, "run_id": run_id, "count": len(items), "items": items})


@app.route('/api/v1/db/feedback', methods=['GET'])
def feedback_summary():
    """Return per-indicator feedback counts from the database."""
    if not db:
        return jsonify({"success": False, "error": "Database not available"}), 503
    counts = db.get_feedback_counts()
    return jsonify({"success": True, "count": len(counts), "feedback": counts})


@app.route('/api/v1/db/model-registry', methods=['GET'])
def model_registry():
    """Return all registered model versions from the database."""
    if not db:
        return jsonify({"success": False, "error": "Database not available"}), 503
    versions = db.get_model_versions()
    return jsonify({"success": True, "count": len(versions), "versions": versions})
