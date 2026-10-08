# Model

## Fine-Tuned Model

The proposed Hausa sentiment classifier is based on:

`xlm-roberta-base`

The model was fine-tuned for three-class sentiment classification:

- Negative
- Neutral
- Positive

The maximum input sequence length used during tokenization was 128 tokens.

## Training

The model-development process was performed in Google Colab using GPU acceleration.

The research workflow included:

- Baseline development
- Transformer fine-tuning
- Controlled learning-rate experiments
- Validation evaluation
- Final test evaluation
- Error analysis

## Model Storage

The fine-tuned model is excluded from this GitHub repository because Transformer model files can be large.

The deployment version of the model will be hosted separately and loaded by the web application.

The final trained model directory is:

`xlmr_hausa_sentiment`