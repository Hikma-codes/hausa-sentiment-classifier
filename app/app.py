from predict import predict_sentiment
from model_utils import load_model
import sys
from pathlib import Path

import streamlit as st


# Allow the app to import modules from the src directory
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

sys.path.append(str(SRC_DIR))


# --------------------------------------------------
# Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="HausaSentiment",
    page_icon="💬",
    layout="centered",
)


MODEL_PATH = ROOT_DIR / "models" / "xlmr_hausa_sentiment"


# --------------------------------------------------
# Page title
# --------------------------------------------------

st.title("HausaSentiment")

st.subheader("Hausa Sentiment Classification")

st.write(
    "Enter a Hausa text and the fine-tuned XLM-R model "
    "will classify its sentiment as negative, neutral, or positive."
)


# --------------------------------------------------
# Load model
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


try:
    tokenizer, model, device = get_model()

except FileNotFoundError:
    st.error(
        "The trained model was not found. "
        "Please place the fine-tuned XLM-R model inside "
        "'models/xlmr_hausa_sentiment'."
    )
    st.stop()

except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


# --------------------------------------------------
# User input
# --------------------------------------------------

text = st.text_area(
    "Enter Hausa text",
    placeholder="Rubuta magana a Hausa a nan...",
    height=150,
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button("Analyze Sentiment", type="primary"):

    if not text.strip():
        st.warning("Please enter some Hausa text.")

    else:

        try:

            result = predict_sentiment(
                text=text,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )

            label = result["label"]
            confidence = result["confidence"]

            st.success(f"Sentiment: {label}")

            st.metric(
                "Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.write("### Class probabilities")

            for sentiment, probability in result["probabilities"].items():

                st.write(
                    f"**{sentiment}**: "
                    f"{probability * 100:.2f}%"
                )

                st.progress(probability)

        except Exception as error:
            st.error(f"Prediction failed: {error}")


# --------------------------------------------------
# Project information
# --------------------------------------------------

st.divider()

st.caption(
    "HausaSentiment — Transformer-based sentiment "
    "classification for Hausa text using fine-tuned XLM-R."
)
