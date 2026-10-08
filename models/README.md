# Model Files

The fine-tuned XLM-R model used by HausaSentiment is stored separately from the GitHub source repository because trained model files are large.

The model used in this project is based on:

- `xlm-roberta-base`
- Three sentiment classes:
  - Negative
  - Neutral
  - Positive
- Maximum sequence length: 128 tokens

The final fine-tuned model is named:

`xlmr_hausa_sentiment`

The model is loaded by the Streamlit application during inference.