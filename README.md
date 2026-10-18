HausaSentiment: Hausa Language Sentiment Classification
An end-to-end Natural Language Processing (NLP) project for classifying Hausa text into Negative, Neutral, and Positive sentiment using traditional machine learning and a fine-tuned XLM-RoBERTa transformer model.



1. Project Overview
HausaSentiment is a Natural Language Processing project that investigates how machine learning can be used to identify sentiment in written Hausa text.
The project focuses on three sentiment categories:
Negative: Text expressing dissatisfaction, criticism, disappointment, or another negative sentiment.
Neutral: Text that is primarily factual or does not clearly express positive or negative sentiment.
Positive: Text expressing satisfaction, approval, happiness, or another positive sentiment.
The project compares two approaches to text classification:
TF-IDF with Logistic Regression: A traditional machine learning baseline that converts text into numerical features and uses those features to predict sentiment.
Fine-tuned XLM-RoBERTa: A multilingual transformer model adapted to the Hausa sentiment classification task using labelled Hausa text.
The project follows an end-to-end machine learning workflow, beginning with dataset preparation and exploratory data analysis, continuing through model development and evaluation, and ending with deployment of an interactive web application.
The final application allows users to enter Hausa text and receive a predicted sentiment category together with estimated class probabilities.
Rather than treating model training as the entire project, HausaSentiment also considers evaluation, error analysis, reproducibility, deployment, and the practical limitations of applying NLP to a language with fewer digital resources than many widely supported languages.
2. Project Links
Resource
Link
GitHub repository
View the source code
Live Streamlit application
Try HausaSentiment
Project demonstration video
Watch the YouTube demo
Fine-tuned model on Hugging Face
View the model repository
AfriSenti dataset and project resources
AfriSenti GitHub repository
SemEval-2023 Task 12
Sentiment Analysis for African Languages

3. Problem Statement and Motivation
Sentiment analysis is an NLP task that identifies opinions, attitudes, and emotional polarity expressed in text. It can support applications such as customer feedback analysis, public opinion research, social media monitoring, and the analysis of user-generated content.
However, the effectiveness of sentiment analysis depends on the availability of suitable data, appropriate language representations, and models that can interpret the text being analysed.
Many widely used NLP tools and pretrained models have been developed around languages with substantial digital resources. For languages such as Hausa, access to labelled datasets and well-evaluated language technologies remains an important consideration.
Hausa is widely spoken across West Africa and beyond. Improving tools for analysing Hausa text can contribute to more inclusive language technology and create opportunities for practical NLP applications.
The central problem addressed by this project is:
How effectively can traditional machine learning and a fine-tuned multilingual transformer model classify Hausa text into negative, neutral, and positive sentiment categories?
The project investigates this question by developing a traditional machine learning baseline, fine-tuning XLM-RoBERTa, comparing their performance, analysing prediction errors, and deploying the selected model in a web application.
The project is an experimental implementation rather than a claim that automated sentiment classification can fully capture every cultural, linguistic, or contextual meaning in Hausa.
4. Dataset
4.1 Data source
The project uses Hausa sentiment data from the AfriSenti project.
AfriSenti is a research initiative focused on sentiment analysis for African languages. Its resources were developed to support research and evaluation of sentiment classification in languages that are underrepresented in many NLP datasets.
The dataset provides labelled text examples that can be used to train and evaluate supervised sentiment classification models.
Dataset source: AfriSenti GitHub repository
Related research paper: AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages
4.2 Dataset splits
The project uses the available training, validation/development, and test splits for model development and evaluation.
Training set: Used to fit the baseline and fine-tune the transformer model.
Validation/development set: Used to compare learning-rate experiments and select the model configuration.
Test set: Used for final evaluation after the model configuration has been selected.
The test set should not be used repeatedly to select hyperparameters. Keeping it separate from model selection provides a more meaningful assessment of performance on held-out examples.
4.3 Dataset considerations
Social media text can contain spelling variations, informal expressions, abbreviations, short messages, and context-dependent meanings. These characteristics can make sentiment classification challenging.
The dataset also represents a particular collection of labelled examples. Performance on this dataset does not automatically guarantee equivalent performance on every Hausa-speaking community, writing style, dialect, domain, or real-world application.
5. Technology Stack
The project combines traditional machine learning, transformer-based NLP, data analysis, and web application deployment.
Technology
Purpose
Python
Main programming language
pandas
Data loading and tabular data processing
NumPy
Numerical operations
scikit-learn
TF-IDF, Logistic Regression, and classification evaluation
PyTorch
Deep learning model execution
Hugging Face Transformers
Loading and fine-tuning XLM-RoBERTa
Hugging Face Tokenizers
Efficient text tokenization
SentencePiece
Tokenization support for the transformer
Matplotlib / Seaborn, where used
Exploratory plots and evaluation visualizations
Jupyter Notebook / Google Colab
Experimentation and model development
Streamlit
Interactive prediction interface
Hugging Face Hub
Hosting the fine-tuned model and tokenizer
Streamlit Community Cloud
Hosting the web application
Git and GitHub
Version control and source-code hosting

The actual versions used for deployment are documented in requirements.txt. Keeping deployment dependencies consistent is particularly important when using pretrained transformer models.
6. Project Workflow
The project follows this general workflow:
Data acquisition: Obtain the Hausa sentiment dataset from AfriSenti.
Data inspection: Examine the available splits, label distribution, missing values, duplicates, and text characteristics.
Preprocessing: Apply the selected data-cleaning and validation steps.
Exploratory data analysis: Visualize sentiment distribution and text-length characteristics.
Baseline development: Convert text to TF-IDF features and train Logistic Regression.
Transformer fine-tuning: Fine-tune XLM-RoBERTa for the three-class classification task.
Hyperparameter comparison: Evaluate the selected learning rates on the validation/development set.
Model selection: Choose the configuration according to validation weighted F1-score.
Final evaluation: Evaluate the selected model on the held-out test set.
Error analysis: Inspect incorrect predictions and possible explanations.
Model publication: Host the fine-tuned model and tokenizer on Hugging Face.
Application development: Build an interface that accepts Hausa text and returns sentiment predictions.
Deployment: Publish the Streamlit application and test the live inference workflow.
7. Data Preparation and Preprocessing
Data preparation is important because inconsistent labels, missing text, duplicates, and unsuitable input formats can affect training and evaluation.
The project includes checks and preprocessing steps for preparing the available data.
7.1 Data inspection
The dataset is inspected to understand its structure, available labels, text fields, and data splits.
Important checks include:
Confirming that the expected text and label columns are present.
Checking for missing or empty text.
Checking the available sentiment classes.
Inspecting duplicate records.
Reviewing the number of examples in each split.
Examining text length and the distribution of sentiment labels.
7.2 Baseline preprocessing
The TF-IDF baseline requires text to be represented numerically. TF-IDF transforms text into a sparse feature matrix based on term frequency and inverse document frequency.
The vectorizer configuration determines how words and, where configured, word sequences contribute to the representation. The resulting feature matrix is used to train Logistic Regression.
7.3 Transformer preprocessing
XLM-RoBERTa uses its pretrained tokenizer to convert text into token IDs and attention masks.
The experiment uses a maximum sequence length of 128 tokens. Longer inputs are handled according to the truncation configuration, while shorter inputs are padded as required by the model.
The model's tokenizer must match the fine-tuned model checkpoint. Using a different tokenizer or an inconsistent label mapping could result in incorrect inference.
7.4 Inference-time preprocessing
The deployed application strips leading and trailing whitespace and converts the input text to lowercase before tokenization.
This should be distinguished from the complete training-data preprocessing pipeline. Lowercasing at inference time is an explicit application behaviour; it should not be interpreted as evidence that every training preprocessing operation is repeated in the deployed application.
8. Models and Experiments
8.1 Model 1: TF-IDF with Logistic Regression
The first model is a traditional machine learning baseline.
TF-IDF represents text using numerical features based on the importance of terms within documents and across the corpus. Logistic Regression learns a decision boundary that maps these features to the three sentiment classes.
The baseline provides a useful point of comparison for the transformer model.
Advantages of this approach include:
Relatively straightforward implementation.
A useful benchmark for supervised text classification.
Lower computational requirements than fine-tuning a large transformer.
A feature representation that can be inspected to understand which terms influence predictions.
Limitations include:
Limited ability to represent word meaning in context.
Difficulty handling some linguistic variations and expressions.
Dependence on the selected text features and preprocessing configuration.
8.2 Model 2: Fine-tuned XLM-RoBERTa
The second approach uses XLM-RoBERTa, a multilingual transformer pretrained on text from multiple languages.
Unlike a traditional TF-IDF representation, XLM-RoBERTa produces contextualized text representations. Its pretrained parameters provide a starting point that can be adapted to the Hausa sentiment classification task.
The project fine-tunes the pretrained xlm-roberta-base model using labelled Hausa examples.
The fine-tuned model is available here:
HausaSentiment XLM-RoBERTa model on Hugging Face
8.3 Training configuration
The main experimental settings include:
Parameter
Setting
Base model
xlm-roberta-base
Task
Three-class sequence classification
Maximum sequence length
128 tokens
Number of epochs per learning-rate experiment
3
Learning rates compared
1e-5, 2e-5, 3e-5
Model-selection metric
Validation weighted F1-score
Final evaluation
Held-out test set
Model framework
PyTorch and Hugging Face Transformers

Other training settings, including batch size, random seed, optimizer configuration, and checkpoint behaviour, should be taken from the final notebook configuration when exact reproducibility is required.
8.4 Learning-rate experiments
Learning rate influences how much the model's parameters change during fine-tuning. A learning rate that is too high may lead to unstable updates, while a learning rate that is too low may limit learning within the available training time.
The project compares three learning rates:
1e-5
2e-5
3e-5
The experiments are compared using validation weighted F1-score. The configuration with the highest validation weighted F1-score is selected for final test evaluation.

9. Model Evaluation
The baseline and transformer models are evaluated using classification metrics that provide different perspectives on predictive performance.
9.1 Evaluation metrics
Accuracy
Accuracy measures the proportion of all evaluated examples that receive the correct prediction.
Precision
Precision measures how many predictions assigned to a class were correct. It helps assess the reliability of positive predictions for each sentiment category.
Recall
Recall measures how many of the actual examples belonging to a class were correctly identified.
F1-score
The F1-score combines precision and recall using their harmonic mean.
Weighted F1-score
Weighted F1 calculates the F1-score for each class and averages the scores according to the number of true examples in each class. It accounts for differences in class support, although a single weighted score can still conceal weaker performance on a less frequent class.
9.2 Evaluation visualizations
The project includes evaluation and exploratory visualizations where available, such as:
Sentiment class distribution.
Text-length distribution.
Baseline confusion matrix.
Learning-rate comparison.
Final model confusion matrix.
These plots support the numerical evaluation by showing the distribution of examples, the types of prediction errors, and differences between model configurations.
The exact paths of the figures should match the files committed to the repository.
10. Error Analysis
Overall metrics are useful, but they do not explain every incorrect prediction. Error analysis examines individual examples to identify patterns that may affect model performance.
The project includes an error-analysis output for examining text samples whose predicted labels differ from their true labels.
Possible areas to investigate include:
Ambiguous or context-dependent expressions.
Short messages with limited sentiment information.
Informal spelling and language variation.
Negation and expressions whose sentiment depends on surrounding words.
Sarcasm, irony, or indirect opinions.
Overlapping characteristics between neutral and positive or negative examples.
Domain differences between the evaluation data and the text entered into the application.
These are potential explanations to investigate, not confirmed causes of every error.
A useful error-analysis table includes the original text, true label, predicted label, and, where available, model confidence or class probabilities.
11. Interactive Web Application
HausaSentiment includes an interactive web application built with Streamlit.
Open the live application
The application connects the trained NLP model to a simple interface so users can test sentiment predictions without manually running the training notebook.
11.1 Main functionality
The application allows a user to:
Open the deployed web interface.
Enter Hausa text into the input field.
Submit the text for prediction.
View the predicted sentiment category.
View the estimated class probabilities returned by the model.
The three possible output classes are Negative, Neutral, and Positive.
11.2 Example inputs
The following Hausa sentences can be used as example inputs during testing:
Example 1
Ina matukar son wannan.
Approximate English meaning: “I really like this.”
Example 2
Ban ji dadin wannan ba.
Approximate English meaning: “I did not enjoy this.”
Example 3
An gudanar da taron yau.
Approximate English meaning: “The meeting was held today.”
These examples illustrate the type of input the application accepts. They are not guarantees of specific model outputs. Actual predictions may vary, so the output displayed by the deployed application should be used when documenting a demonstration.
11.3 Important interpretation note
The application returns model predictions, not definitive judgements about human emotions or intentions. The displayed probabilities are model-generated estimates and should not automatically be interpreted as calibrated probabilities or certainty about the true sentiment.
The application is intended as a research and learning prototype rather than a production-grade sentiment moderation or decision-making system.
12. Deployment Architecture
The project separates model training from application inference.
The training workflow is conducted in the model-development environment. After training and evaluation, the selected model and tokenizer are published to Hugging Face. The Streamlit application loads the saved model artefacts to make predictions.
The deployment uses the following services:
Component
Role
GitHub
Hosts the application source code and project documentation
Hugging Face Hub
Hosts the fine-tuned model and tokenizer
Streamlit Community Cloud
Runs and serves the interactive web application
User's browser
Provides the interface for submitting text and viewing predictions

12.1 Model loading
The application is configured to use the local model directory when the expected model directory is available. Otherwise, it loads the model from the public Hugging Face repository:
https://huggingface.co/hhiikkmaa/hausa-sentiment-xlmr
This arrangement avoids requiring the large model weights to be committed to the GitHub source-code repository. It also avoids repeating the training process whenever a user submits text through the application.
12.2 Application deployment
The Streamlit application source is located at:
app/app.py
The deployed application is available at:
https://hausa-sentiment-classifier-zpzfzuovfjnxluqptxtyoo.streamlit.app/
The dependency list is maintained in requirements.txt so the deployment environment can install the libraries required by the application.
13. Deployment Challenges and Solutions
Building the model was only one part of the project. Making the model accessible through a public web application introduced additional engineering challenges.
13.1 Hosting large model files
Transformer model weights can be large. Committing large model files directly to a source-code repository can make version control and deployment more difficult.
Solution: The fine-tuned model and tokenizer were hosted on Hugging Face, while the GitHub repository was used for application code and project files.
13.2 Application import paths
The application initially encountered an import-path issue when deployed because the cloud environment could not resolve an internal project module.
Solution: The application was adjusted so the inference logic could operate directly from the application file, avoiding reliance on the problematic internal import path.
13.3 Dependency compatibility
Machine learning libraries depend on particular versions of other packages. A combination that works in one environment may fail when installed in another.
Solution: Compatible package versions were specified in requirements.txt, including pinned versions for important dependencies such as Transformers, Tokenizers, and Hugging Face Hub.
13.4 Separating training from inference
Training a transformer model is computationally more demanding than using an already trained model to make predictions.
Solution: The deployed application loads the saved model and tokenizer for inference rather than training the model every time the application starts or a user submits text.
13.5 Testing the complete workflow
A successful local run does not automatically mean that a cloud deployment will work correctly. The application must also be able to install dependencies, locate its model, load the tokenizer, accept text, and return predictions in the hosted environment.
Lesson learned: Model development, packaging, dependency management, hosting, and end-to-end testing are all important parts of an applied machine learning project.

15. Installation and Local Setup
Follow these steps to run the Streamlit application locally.
15.1 Prerequisites
Install or prepare the following:
Python compatible with the versions supported by the project's dependencies.
Git.
A terminal such as Git Bash, PowerShell, or Command Prompt.
Internet access for downloading dependencies and, when no local model is present, loading the model from Hugging Face.
15.2 Clone the repository
Open a terminal and run:
git clone https://github.com/Hikma-codes/hausa-sentiment-classifier.git
Move into the project directory:
cd hausa-sentiment-classifier
15.3 Create a virtual environment
On Windows, run:
python -m venv .venv
Activate the environment in PowerShell:
.\.venv\Scripts\Activate.ps1
Or activate it in Git Bash:
source .venv/Scripts/activate
15.4 Install dependencies
With the virtual environment activated, run:
python -m pip install --upgrade pip
pip install -r requirements.txt
Using the project's requirements file helps maintain consistency with the deployment environment.
15.5 Launch the application
From the root of the repository, run:
streamlit run app/app.py
Streamlit should display a local address in the terminal, usually:
http://localhost:8501
Open the displayed address in your browser.
15.6 Model availability
The application uses the local model directory if the expected directory is present. Otherwise, it attempts to load the public model from Hugging Face.
The hosted model is available at:
https://huggingface.co/hhiikkmaa/hausa-sentiment-xlmr
The first model load may take time because the required model files may need to be downloaded. Available memory, network speed, and the installed PyTorch environment can also affect startup time.
16. Using the Application
Once the application is open:
Enter a Hausa sentence or short text.
Submit the input using the interface.
Wait for inference to complete.
Read the predicted sentiment category.
Review the estimated probabilities for the available classes.
For meaningful testing, try inputs with different sentiment characteristics and compare the predictions with your own interpretation of the text.
It is useful to test different sentence lengths and expressions rather than relying on one example. However, predictions should be treated cautiously when the input contains sarcasm, ambiguous wording, unfamiliar spelling, or context that is not available to the model.
For a guided demonstration, watch the HausaSentiment project video.
17. Reproducing the Experiments
The training notebook is the main starting point for understanding and reproducing the model-development workflow.
Notebook: notebooks/Hausa_Sentiment_Classification.ipynb
To reproduce the experiments:
Clone the repository.
Obtain the AfriSenti Hausa dataset from the official project repository.
Follow the dataset preparation instructions documented in the project.
Open the notebook in Jupyter or Google Colab.
Configure the dataset paths for the environment being used.
Run the data inspection and preprocessing steps.
Train and evaluate the TF-IDF and Logistic Regression baseline.
Fine-tune XLM-RoBERTa using the documented experimental settings.
Compare the validation results for the three learning rates.
Evaluate the selected configuration on the held-out test set.
Review the evaluation plots and error-analysis output.

Reproducibility considerations
Exact numerical results can vary when the runtime environment, random seeds, library versions, hardware, or training configuration differ.
For a complete reproducibility record, document the dataset version, split sizes, preprocessing configuration, model checkpoint, random seed, hyperparameters, package versions, and evaluation procedure.
The model used for the deployed application is available on Hugging Face. Running the application does not require rerunning the training notebook unless the goal is to reproduce or modify the experiments.
18. Limitations
HausaSentiment demonstrates a practical NLP workflow, but it has several limitations.
18.1 Dataset coverage
The model is trained and evaluated using the selected AfriSenti Hausa dataset. Its performance may differ on text from other sources, domains, communities, or time periods.
18.2 Language variation
Hausa text can vary in spelling, vocabulary, phrasing, and writing style. These differences may affect how well the model interprets an input.
18.3 Context and ambiguity
A sentence may express different sentiments depending on the surrounding conversation or the speaker's intention. A classifier that receives only a short text may not have enough information to resolve such ambiguity.
18.4 Sarcasm and indirect sentiment
Sarcasm, irony, humour, and indirect expressions can be difficult for automated sentiment classifiers because the literal wording may not match the intended sentiment.
18.5 Model confidence
The class probabilities displayed by the application are model outputs. They should not be treated as a guarantee that a prediction is correct or as calibrated estimates of real-world certainty.
18.6 Generalization
Evaluation scores on a held-out dataset provide useful evidence about performance under the tested conditions. They do not prove that the model will perform equally well in every real-world setting.
18.7 Application scope
The deployed application is a prototype for demonstrating sentiment classification. It has not been established as a production-grade system for high-stakes decisions, automated moderation, or decisions about individuals.
19. Future Improvements
Potential future work includes:
Expanded evaluation: Test the model on additional Hausa datasets and text from different sources.
Dialect and spelling robustness: Investigate how spelling variations and regional language differences affect predictions.
Class-level evaluation: Examine precision, recall, and F1-score for each sentiment class rather than relying only on aggregate metrics.
Error-analysis improvements: Group misclassifications by identifiable patterns and evaluate whether targeted preprocessing or training changes help.
Model comparison: Explore additional multilingual models or alternative text-classification approaches.
Calibration analysis: Assess whether the probabilities displayed by the model align with observed prediction correctness.
Application usability: Improve the input experience, error handling, and presentation of model outputs.
Performance optimization: Investigate model loading time, inference latency, and resource use.
Reproducibility: Document exact dataset counts, software versions, random seeds, and final experiment outputs.
Broader language technology: Explore comparable approaches for other African languages when suitable datasets and evaluation resources are available.
These are possible extensions rather than features already implemented in the current version.
20. Learning Outcomes
This project provided practical experience with the stages involved in developing and deploying an NLP system.
Key areas of work included:
Preparing labelled text for supervised learning.
Exploring a language-specific sentiment dataset.
Developing a traditional machine learning baseline.
Fine-tuning a multilingual transformer model.
Comparing hyperparameters using a validation metric.
Evaluating classification performance.
Investigating misclassified examples.
Saving and hosting model artefacts.
Building a user-facing application with Streamlit.
Managing dependencies and deployment configuration.
Debugging import paths and cloud-environment issues.
Connecting model development, version control, model hosting, and deployment into one workflow.
The project illustrates that applied machine learning involves more than choosing an algorithm. Data preparation, evaluation, software engineering, deployment, and documentation all contribute to a usable and understandable system.
21. Acknowledgements
This project uses research resources from the AfriSenti initiative and builds on the work of researchers developing NLP datasets and methods for African languages.
Thanks to the researchers and contributors who made the dataset and associated research publicly available.
The project also uses open-source tools from the Python, scikit-learn, PyTorch, Hugging Face, and Streamlit ecosystems.
23. References
Muhammad, S. H., et al. (2023). AfriSenti: A Twitter Sentiment Analysis Benchmark for African Languages. Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing.
Paper: https://aclanthology.org/2023.emnlp-main.862/
Muhammad, S. H., et al. (2023). SemEval-2023 Task 12: Sentiment Analysis for African Languages (AfriSenti-SemEval). Proceedings of the 17th International Workshop on Semantic Evaluation.
Paper: https://aclanthology.org/2023.semeval-1.315/
Conneau, A., et al. (2020). Unsupervised Cross-lingual Representation Learning at Scale. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics.
Paper: https://aclanthology.org/2020.acl-main.747/
Wolf, T., et al. (2020). Transformers: State-of-the-Art Natural Language Processing. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations.
Paper: https://aclanthology.org/2020.emnlp-demos.6/
AfriSenti Project. Dataset, code, and research resources for sentiment analysis in African languages.
Repository: https://github.com/afrisenti-semeval/afrisent-semeval-2023
Hugging Face Transformers. Documentation for pretrained transformer models and sequence classification.
Documentation: https://huggingface.co/docs/transformers/
Hugging Face Hub. Model hosting and sharing platform.
Documentation: https://huggingface.co/docs/hub/
Streamlit. Documentation for building Python data applications.
Documentation: https://docs.streamlit.io/
24. Author
Hikma Hamza
BSc Software Engineering
African Leadership University
Project repository: HausaSentiment
For a walkthrough of the application and project workflow, watch the HausaSentiment demonstration video.

HausaSentiment is an educational and experimental NLP project intended to explore sentiment classification for Hausa text. Predictions are model-generated estimates and should be interpreted with the limitations of the dataset and model in mind.
