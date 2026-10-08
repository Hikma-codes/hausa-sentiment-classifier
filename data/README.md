# Dataset

## AfriSenti Hausa Dataset

HausaSentiment uses the Hausa portion of the AfriSenti benchmark for three-class sentiment classification.

The dataset contains naturally occurring Hausa social media text annotated with three sentiment labels:

- Negative
- Neutral
- Positive

## Data Splits

The official dataset splits are preserved:

- Training set
- Development/validation set
- Test set

The research notebook loads these splits separately and uses them for model training, validation, and final evaluation.

## Preprocessing

The research workflow performs basic data-quality checks before model training, including:

- Missing-value checks
- Duplicate detection
- Empty-text checks
- Label validation
- Class-distribution analysis
- Text-length analysis

The dataset is not synthetically generated for this project.

## Repository Policy

The raw dataset files are not committed to GitHub.

This repository instead contains the code and documentation required to understand and reproduce the data preparation workflow.