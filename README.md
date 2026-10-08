# HausaSentiment

HausaSentiment is a transformer-based sentiment classification system for Hausa text.

The project investigates whether a multilingual pretrained transformer can improve Hausa sentiment classification compared with a traditional TF-IDF and Logistic Regression baseline.

## Project Objective

The goal is to classify Hausa text into three sentiment categories:

- Negative
- Neutral
- Positive

The project uses the Hausa portion of the AfriSenti benchmark, which contains naturally occurring Hausa social media text.

## Methodology

The research workflow includes:

1. Dataset preparation and quality analysis
2. Text preprocessing
3. TF-IDF + Logistic Regression baseline
4. Fine-tuning XLM-RoBERTa
5. Controlled learning-rate experiments
6. Final model evaluation
7. Confusion-matrix analysis
8. Error analysis
9. Web application deployment

## Model

The proposed model is `xlm-roberta-base`, fine-tuned for three-class Hausa sentiment classification.

The model was trained using the official train, development, and test splits of the Hausa AfriSenti dataset.

## Repository Structure

```text
hausa-sentiment-classifier/
│
├── app/
│   └── app.py
│
├── src/
│   ├── preprocessing.py
│   ├── predict.py
│   └── model_utils.py
│
├── notebooks/
│   └── Hausa_Sentiment_Classification.ipynb
│
├── results/
│   ├── figures/
│   └── tables/
│
├── data/
│   └── README.md
│
├── models/
│   └── README.md
│
├── requirements.txt
├── README.md
├── .gitignore

Experiments
The project compares a traditional TF-IDF + Logistic Regression baseline with the fine-tuned XLM-R model.
Controlled learning-rate experiments were also performed while keeping the other major training conditions fixed.
The experiment results and visualizations are available in the results/ directory.
Dataset
The project uses the Hausa portion of the AfriSenti benchmark.
The raw dataset is not committed to this repository. See data/README.md for information about the dataset and its organization.
Model Files
The trained model is intentionally excluded from GitHub because transformer model files can be large.
See models/README.md for information about the trained model.
Application
The final system will provide a simple web interface where a user can enter Hausa text and receive:
- Predicted sentiment
- Prediction confidence
- Probability for each sentiment class
The application is implemented using Streamlit.
Reproducibility
The research notebook contains the complete model-development workflow, including dataset preparation, baseline development, transformer fine-tuning, controlled experiments, evaluation, and error analysis.
Technologies
- Python
- PyTorch
- Hugging Face Transformers
- XLM-RoBERTa
- Scikit-learn
- Streamlit
- Google Collab
- Git and GitHub