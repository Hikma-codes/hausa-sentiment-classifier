import torch

from model_utils import LABELS
from preprocessing import clean_text


def predict_sentiment(
    text,
    tokenizer,
    model,
    device,
    max_length=128,
):
    """
    Predict the sentiment of a Hausa text.
    """

    text = clean_text(text)

    if not text:
        raise ValueError(
            "Please enter some Hausa text."
        )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1,
    )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()

    confidence = probabilities[
        predicted_class
    ].item()

    return {
        "label": LABELS[predicted_class],
        "confidence": confidence,
        "probabilities": {
            LABELS[i]: probabilities[i].item()
            for i in range(len(LABELS))
        },
    }
