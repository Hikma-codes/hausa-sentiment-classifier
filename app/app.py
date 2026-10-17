
from pathlib import Path

import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# Project paths and hosted model
ROOT_DIR = Path(__file__).resolve().parent.parent
LOCAL_MODEL_PATH = ROOT_DIR / "models" / "xlmr_hausa_sentiment"
MODEL_ID = "hhiikkmaa/hausa-sentiment-xlmr"

# Use the local model on your computer and Hugging Face in the cloud.
MODEL_SOURCE = (
    str(LOCAL_MODEL_PATH)
    if LOCAL_MODEL_PATH.is_dir()
    else MODEL_ID
)

DEFAULT_LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


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
def get_model(model_source):
    tokenizer = AutoTokenizer.from_pretrained(model_source)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source
    )

    if model.config.num_labels != 3:
        raise ValueError(
            f"Expected 3 sentiment classes, "
            f"but the model has {model.config.num_labels}."
        )

    labels = DEFAULT_LABELS.copy()

    saved_labels = getattr(model.config, "id2label", None)
    if saved_labels:
        converted = {}

        for index, name in saved_labels.items():
            name = str(name).strip().lower()

            if "negative" in name:
                converted[int(index)] = "Negative"
            elif "neutral" in name:
                converted[int(index)] = "Neutral"
            elif "positive" in name:
                converted[int(index)] = "Positive"

        if len(converted) == 3:
            labels = converted

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device, labels


try:
    tokenizer, model, device, labels = get_model(MODEL_SOURCE)

except Exception as error:
    st.error("The model could not be loaded.")
    st.write("Model source:")
    st.code(MODEL_SOURCE)
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

if st.button(
    "Analyze sentiment",
    type="primary",
    use_container_width=True,
):
    cleaned_text = text_input.strip().lower()

    if not cleaned_text:
        st.warning("Please enter some text first.")

    else:
        try:
            with st.spinner("Analyzing your text..."):
                inputs = tokenizer(
                    cleaned_text,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=128,
                )

                inputs = {
                    name: tensor.to(device)
                    for name, tensor in inputs.items()
                }

                with torch.no_grad():
                    outputs = model(**inputs)

                probabilities = torch.softmax(
                    outputs.logits, dim=-1
                )[0]

                predicted_index = int(
                    torch.argmax(probabilities).item()
                )

                predicted_label = labels[predicted_index]
                confidence = float(
                    probabilities[predicted_index].item()
                )

            st.subheader("Predicted sentiment")
            st.header(predicted_label)

            st.write(
                f"Prediction confidence: {confidence:.1%}"
            )
            st.progress(confidence)

            st.subheader("Class probabilities")

            for index, label in labels.items():
                probability = float(probabilities[index].item())
                st.write(f"**{label}: {probability:.1%}**")
                st.progress(probability)

            st.caption(
                "Confidence represents the model's estimated "
                "probability, not a guarantee that the prediction "
                "is correct."
            )

        except Exception as error:
            st.error("Prediction failed.")
            st.code(f"{type(error).__name__}: {error}")


st.divider()
st.caption("HausaSentiment | Hausa-language NLP project")
