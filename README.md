HausaSentiment
A Hausa-language sentiment classifier that predicts whether a text expresses Negative, Neutral, or Positive sentiment.
Project links
- Live app: https://hausa-sentiment-classifier-zpzfzuovfjnxluqptxtyoo.streamlit.app/
- Demo video: https://www.youtube.com/watch?v=j7CMh5eaS2I
- GitHub repository: https://github.com/Hikma-codes/hausa-sentiment-classifier
- Fine-tuned model: https://huggingface.co/hhiikkmaa/hausa-sentiment-xlmr
- Dataset and resources (AfriSenti): https://github.com/afrisenti-semeval/afrisent-semeval-2023


Overview
HausaSentiment is an end-to-end Natural Language Processing (NLP) project focused on classifying written Hausa text into three sentiment categories:
- Negative
- Neutral
- Positive

The project compares a traditional machine-learning baseline with a fine-tuned multilingual transformer model. The selected model is available through a Streamlit web application.
Models
Two approaches were explored:
1. TF-IDF + Logistic Regression — a traditional machine-learning baseline that represents text using TF-IDF features and predicts a sentiment class.

2. XLM-RoBERTa — a multilingual transformer fine-tuned for Hausa sentiment classification.
The transformer experiments compared learning rates of 1e-5, 2e-5, and 3e-5. The final configuration was selected using validation weighted F1-score. Refer to the project notebook and report for the experimental results and detailed analysis.


Dataset
The project uses Hausa sentiment data from the AfriSenti project.
The dataset contains labelled Hausa text used for training, validation, and testing. The data splits and label distribution are documented in the notebook and report.
Technology
- Python
- pandas and NumPy
- scikit-learn
- PyTorch
- Hugging Face Transformers and Tokenizers
- SentencePiece
- Streamlit
- Hugging Face Hub
- Git and GitHub


Try the application
Open the HausaSentiment live app, enter a Hausa sentence or short text, and submit it to view the predicted sentiment and model-estimated class probabilities.
The probabilities are model outputs and should not be interpreted as guaranteed measures of correctness.


Run locally
1. Clone the repository
git clone https://github.com/Hikma-codes/hausa-sentiment-classifier.git
cd hausa-sentiment-classifier
2. Create and activate a virtual environment
Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
Windows (Git Bash):
python -m venv .venv
source .venv/Scripts/activate
macOS or Linux:
python3 -m venv .venv
source .venv/bin/activate
3. Install dependencies
pip install -r requirements.txt
4. Start the Streamlit app
streamlit run app/app.py
The application uses the model files in the local project directory when they are available. Otherwise, it loads the published model from Hugging Face.


Project structure
The main components include:
- app/app.py — Streamlit application for sentiment prediction.
- requirements.txt — Python dependencies.
- The project notebook — data preparation, model training, evaluation, and analysis.
- The project report — methodology, results, discussion, and limitations.
Check the repository for the exact locations of the notebook and report.


Limitations
- The model's performance may vary across Hausa dialects, spelling styles, topics, and text sources.
- Short texts may lack context or contain sarcasm, mixed sentiment, or indirect meanings.
- The model can make incorrect predictions, including when its estimated probability is high.
- Evaluation on the selected dataset does not guarantee the same performance in every real-world setting.
- The application is a project prototype and should not be used as the sole basis for high-stakes decisions.


References
- AfriSenti: https://github.com/afrisenti-semeval/afrisent-semeval-2023
- SemEval-2023 Task 12 — Sentiment Analysis for African Languages: https://aclanthology.org/2023.semeval-1.315/
- AfriSenti paper: https://aclanthology.org/2023.emnlp-main.862/
- XLM-RoBERTa paper: https://aclanthology.org/2020.acl-main.747/
- Transformers library paper: https://aclanthology.org/2020.emnlp-demos.6/


Author
Hikma Hamza
- GitHub: https://github.com/Hikma-codes
- Project repository: https://github.com/Hikma-codes/hausa-sentiment-classifier