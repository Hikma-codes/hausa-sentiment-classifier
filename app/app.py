
from src.predict import predict_sentiment
from src.model_utils import load_model
import sys
from pathlib import Path

import streamlit as st


# Find the project root, regardless of where app.py is located.
ROOT_DIR = Path(__file__).resolve().parent.parent

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

MODEL_PATH = ROOT_DIR / "models" / "xlmr_hausa_sentiment"

# Import only after adding the project root to Python's path.


st.set_page_config(
    page_title="HausaSentiment",
    page_icon="💬",
    layout="centered",
)

st.title("HausaSentiment")

st.write(
    "A transformer-based sentiment classifier for Hausa text. "
    "Enter a Hausa sentence or social media message to classify "
    "its sentiment as Negative, Neutral, or Positive."
)

st.caption("Powered by your fine-tuned XLM-RoBERTa model.")


@st.cache_resource(show_spinner="Loading your trained model...")
def get_model(model_directory):
    return load_model(model_directory)


# Load model and show useful errors rather than hiding the traceback.
try:
    tokenizer, model, device, labels = get_model(str(MODEL_PATH))

except Exception as error:
    st.error("The model could not be loaded.")

    st.write("Expected model location:")
    st.code(str(MODEL_PATH))

    st.write("Error details:")
    st.code(f"{type(error).__name__}: {error}")

    st.stop()


st.success(f"Model loaded successfully on {device}.")

st.subheader("Analyze Hausa text")

text_input = st.text_area(
    "Enter your Hausa sentence or message",
    placeholder="Ina matukar son wannan.",
    height=130,
)

if st.button("Analyze sentiment", type="primary", use_container_width=True):
    if not text_input.strip():
        st.warning("Please enter some text first.")

    else:
        try:
            with st.spinner("Analyzing your text..."):
                result = predict_sentiment(
                    text=text_input,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    labels=labels,
                )

            st.subheader("Predicted sentiment")
            st.header(result["label"])

            st.write(
                f"Prediction confidence: {result['confidence']:.1%}"
            )
            st.progress(result["confidence"])

            st.subheader("Class probabilities")

            for label, probability in result["probabilities"].items():
                st.write(f"**{label}: {probability:.1%}**")
                st.progress(probability)

            st.caption(
                "Confidence represents the model's estimated probability, "
                "not a guarantee that the prediction is correct."
            )

        except Exception as error:
            st.error("Prediction failed.")
            st.code(f"{type(error).__name__}: {error}")


st.divider()
st.caption("HausaSentiment | Hausa-language NLP project")
