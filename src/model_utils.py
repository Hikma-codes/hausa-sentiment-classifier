from pathlib import Path

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def load_model(model_path):
    """
    Load the fine-tuned XLM-R tokenizer and model.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model directory was not found: {model_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path)
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    return tokenizer, model, device
