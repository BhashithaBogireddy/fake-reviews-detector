"""Streamlit web app for the Fake Reviews Detector."""
import json
from pathlib import Path

import streamlit as st

import predictor

APP_DIR = Path(__file__).parent
META_PATH = APP_DIR / "model" / "model_metadata.json"

st.set_page_config(page_title="Fake Reviews Detector", page_icon="🔍", layout="centered")


@st.cache_resource(show_spinner="Loading model...")
def get_model():
    return predictor.load_model()


@st.cache_data
def get_metadata():
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # metadata is optional; the app works without it
        return None


# ---------- Header ----------
st.title("🔍 Fake Reviews Detector")
st.write(
    "Paste a product review and the model estimates how much its wording resembles "
    "**computer-generated** reviews from its training data."
)
st.caption("Final-year project demo · Machine learning: TF-IDF + Logistic Regression (scikit-learn)")

# ---------- Load the model (clear error instead of a crash) ----------
try:
    model = get_model()
except FileNotFoundError:
    st.error(
        "Model file not found. Make sure `model/fake_review_pipeline.joblib` exists "
        "next to `app.py` (see the folder structure in the setup instructions)."
    )
    st.stop()
except Exception as e:  # noqa: BLE001
    st.error(
        f"The model could not be loaded ({type(e).__name__}: {e}). "
        "This usually means the installed scikit-learn version differs from the one used for training. "
        "Install the exact versions listed in requirements.txt."
    )
    st.stop()

# ---------- Input ----------
EXAMPLE = ("I bought this blender for smoothies. It works well on fruit, but it struggles "
           "with ice and the lid is a bit loose. Cleaning is easy though.")

if "review_box" not in st.session_state:
    st.session_state["review_box"] = ""

st.button("Insert example review", on_click=lambda: st.session_state.update(review_box=EXAMPLE))

review = st.text_area(
    "Review text (English)",
    key="review_box",
    height=180,
    max_chars=predictor.MAX_CHARS,
    placeholder="Paste or type a product review here...",
)
st.caption(f"Minimum {predictor.MIN_WORDS} words. Longer reviews give more reliable results.")

# ---------- Predict ----------
if st.button("Predict", type="primary"):
    result = predictor.predict_review(review, pipeline=model)

    if not result["ok"]:
        st.error(result["message"])
    else:
        is_fake = result["label"] == 1
        uncertain = result["confidence_band"] == "uncertain"

        st.subheader("Result")
        if uncertain:
            st.warning(f"**Inconclusive** — the model leans: {result['label_name']}")
        elif is_fake:
            st.error(f"**{result['label_name']}**")
        else:
            st.success(f"**{result['label_name']}**")

        col1, col2 = st.columns(2)
        col1.metric("Model score: resembles generated text", f"{result['prob_fake']:.1%}")
        col2.metric("Model score: resembles original text", f"{result['prob_genuine']:.1%}")
        st.progress(min(max(result["prob_fake"], 0.0), 1.0))
        st.caption(f"Words analysed: {result['n_words']}")

        for w in result["warnings"]:
            st.info(w)

        st.caption(result["disclaimer"])

# ---------- Limitations ----------
st.divider()
with st.expander("⚠️ Limitations — please read", expanded=True):
    st.markdown(
        """
- **A prediction is not proof.** The score shows how closely the wording matches patterns in the training data, not whether someone was actually dishonest.
- **What it was trained on:** computer-generated reviews (reported to come from a GPT-2 model) versus presumed-human reviews from one public dataset. It was **not** tested on paid human fake reviews or on real marketplaces such as Amazon or Flipkart.
- **Short reviews are less reliable.** Accuracy was noticeably lower on very short test reviews.
- **Scores near 50% are inconclusive.** On the test set, accuracy dropped sharply for scores in the 35–65% range.
- **English only.** No product-level or reviewer-level generalisation was tested because the dataset has no such IDs.
- **Do not use this to accuse a real person or remove a real review.**
        """
    )

meta = get_metadata()
if meta:
    with st.expander("About the model"):
        m = meta.get("test_metrics_fake_class", {})
        st.write(f"**Model:** {meta.get('model_type', 'n/a')}")
        st.write(f"**Dataset:** {meta.get('dataset', 'n/a')}")
        if m:
            st.write(
                f"**Held-out test results (fake class):** accuracy {m.get('accuracy')}, "
                f"precision {m.get('precision')}, recall {m.get('recall')}, F1 {m.get('f1')} "
                f"on {meta.get('test_rows')} test reviews. These come from a balanced 50/50 test set "
                "and will not carry over to real platforms, where fakes are rare."
            )
        st.write("**Labels:** 0 = Genuine (original), 1 = Fake (computer-generated)")
