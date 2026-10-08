from predict import predict_sentiment
from model_utils import load_model
import sys
from pathlib import Path

import streamlit as st


# --------------------------------------------------
# Project paths
# --------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"

sys.path.append(str(SRC_DIR))


MODEL_PATH = ROOT_DIR / "models" / "xlmr_hausa_sentiment"


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="HausaSentiment",
    page_icon="💬",
    layout="centered",
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("HausaSentiment")

st.write(
    "A transformer-based sentiment classifier "
    "for Hausa text."
)

st.info(
    "Enter a Hausa sentence or social media message "
    "to classify its sentiment as Negative, Neutral, "
    "or Positive."
)


# --------------------------------------------------
# Model loading
# --------------------------------------------------

@st.cache_resource
def get_model():
    return load_model(MODEL_PATH)


try:
    tokenizer, model, device = get_model()

except FileNotFoundError:
    st.error(
        "The trained model was not found."
    )

    st.write(
        "Expected model location:"
    )

    st.code(
        "models/xlmr_hausa_sentiment"
    )

    st.stop()

except Exception as error:
    st.error(
        f"Unable to load the trained model: {error}"
    )

    st.stop()


# --------------------------------------------------
# Input
# --------------------------------------------------

st.subheader("Enter Hausa text")

text = st.text_area(
    "Hausa text",
    placeholder="Rubuta magana a Hausa a nan...",
    height=160,
    label_visibility="collapsed",
)


# --------------------------------------------------
# Prediction
# --------------------------------------------------

if st.button(
    "Analyze Sentiment",
    type="primary",
    use_container_width=True,
):

    if not text.strip():

        st.warning(
            "Please enter some Hausa text first."
        )

    else:

        try:

            result = predict_sentiment(
                text=text,
                tokenizer=tokenizer,
                model=model,
                device=device,
            )

            st.divider()

            st.subheader("Prediction")

            st.success(
                f"Sentiment: {result['label']}"
            )

            st.metric(
                "Confidence",
                f"{result['confidence'] * 100:.2f}%"
            )

            st.subheader(
                "Class probabilities"
            )

            for (
                sentiment,
                probability,
            ) in result["probabilities"].items():

                percentage = probability * 100

                st.write(
                    f"**{sentiment} — "
                    f"{percentage:.2f}%**"
                )

                st.progress(
                    probability
                )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption(
    "HausaSentiment | XLM-R-based Hausa sentiment classification"
)
