"""Reusable prediction logic for the Fake Reviews Detector."""
import re
from pathlib import Path

import joblib

# Relative to this file, so it works on any machine (no Colab/Drive paths)
MODEL_PATH = Path(__file__).parent / "model" / "fake_review_pipeline.joblib"

LABEL_NAMES = {0: "Genuine (OR, original)", 1: "Fake (CG, computer-generated)"}

MIN_WORDS = 3            # below this: rejected
SHORT_WARN_WORDS = 20    # below this: accepted with a reliability warning
MAX_CHARS = 5000
UNCERTAIN_LOW, UNCERTAIN_HIGH = 0.35, 0.65   # fixed display band, not tuned on test data

DISCLAIMER = ("This is a statistical estimate of how much the text resembles computer-generated "
              "reviews in the training data. It is not proof that a review is fake or genuine.")

_cached_pipeline = None


def load_model(path=MODEL_PATH):
    """Load the saved pipeline once and reuse it."""
    global _cached_pipeline
    if _cached_pipeline is None:
        _cached_pipeline = joblib.load(path)
    return _cached_pipeline


def _error(code, message):
    return {"ok": False, "error_code": code, "message": message}


def predict_review(text, pipeline=None):
    """Return a dict. Check result['ok'] first.

    ok=True  -> label, label_name, prob_fake, prob_genuine, confidence_band, warnings, n_words, disclaimer
    ok=False -> error_code, message
    """
    try:
        if text is None:
            return _error("empty", "No review text was provided.")
        if not isinstance(text, str):
            return _error("bad_type", f"Review must be text, but got {type(text).__name__}.")

        cleaned = re.sub(r"\s+", " ", text).strip()
        if not cleaned:
            return _error("empty", "The review is empty. Please type or paste a review.")
        if not re.search(r"[A-Za-z]", cleaned):
            return _error("no_letters", "The review contains no English letters. This model only supports English text.")

        n_words = len(cleaned.split())
        if n_words < MIN_WORDS:
            return _error("too_short", f"The review has only {n_words} word(s). Please enter at least {MIN_WORDS}; "
                                       "very short text cannot be assessed meaningfully.")
        if len(cleaned) > MAX_CHARS:
            return _error("too_long", f"The review is {len(cleaned)} characters long; the limit is {MAX_CHARS}.")

        pipe = pipeline if pipeline is not None else load_model()
        classes = list(pipe.classes_)
        proba = pipe.predict_proba([cleaned])[0]
        prob_fake = float(proba[classes.index(1)])
        prob_genuine = float(proba[classes.index(0)])
        label = int(pipe.predict([cleaned])[0])

        warnings = []
        if n_words < SHORT_WARN_WORDS:
            warnings.append(f"Short review ({n_words} words): the model is less reliable on short text.")
        band = "uncertain" if UNCERTAIN_LOW <= prob_fake <= UNCERTAIN_HIGH else "confident"
        if band == "uncertain":
            warnings.append("The score is close to the decision boundary, so treat this as inconclusive.")

        return {
            "ok": True,
            "label": label,
            "label_name": LABEL_NAMES[label],
            "prob_fake": prob_fake,
            "prob_genuine": prob_genuine,
            "confidence_band": band,
            "n_words": n_words,
            "warnings": warnings,
            "disclaimer": DISCLAIMER,
        }

    except FileNotFoundError:
        return _error("model_missing", f"Model file not found at {MODEL_PATH}. Make sure the 'model' folder is deployed next to predictor.py.")
    except Exception as e:  # noqa: BLE001
        return _error("model_error", f"Prediction failed ({type(e).__name__}: {e}). "
                                     "If this happened after deployment, check that scikit-learn matches requirements.txt.")
