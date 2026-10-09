
import torch

from src.preprocessing import clean_text


def predict_sentiment(
    text,
    tokenizer,
    model,
    device,
    labels,
    max_length=128,
):
    text = clean_text(text)

    if not text:
        raise ValueError("Please enter some Hausa text.")

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    inputs = {
        name: tensor.to(device)
        for name, tensor in inputs.items()
    }

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = torch.softmax(outputs.logits, dim=-1)[0]
    predicted_index = int(torch.argmax(probabilities).item())

    return {
        "label": labels[predicted_index],
        "confidence": float(probabilities[predicted_index].item()),
        "probabilities": {
            labels[index]: float(probabilities[index].item())
            for index in range(len(probabilities))
        },
    }
