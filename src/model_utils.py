
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


LABELS = {
    0: "Negative",
    1: "Neutral",
    2: "Positive",
}


def load_model(model_path):
    model_path = Path(model_path).resolve()

    if not model_path.is_dir():
        raise FileNotFoundError(
            f"Model directory not found: {model_path}"
        )

    if not (model_path / "config.json").exists():
        raise FileNotFoundError(
            f"config.json is missing from {model_path}"
        )

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(model_path),
        local_files_only=True,
    )

    if model.config.num_labels != 3:
        raise ValueError(
            f"Expected 3 sentiment classes, got "
            f"{model.config.num_labels}."
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model.to(device)
    model.eval()

    # Use the saved class names when they are meaningful.
    saved_labels = getattr(model.config, "id2label", None)
    labels = LABELS.copy()

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

    return tokenizer, model, device, labels
