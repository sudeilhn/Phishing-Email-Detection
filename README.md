
# Phishing Email Detection Using Deep Learning

A deep learning project focused on detecting phishing emails in Turkish-language email data using Natural Language Processing (NLP).

The project explores and compares **LSTM, Bidirectional LSTM, CNN-LSTM, and GRU** architectures to evaluate their effectiveness in email classification.

## Overview

The goal is to distinguish phishing emails from legitimate messages through deep learning-based text classification.

- Developed and evaluated 8 model versions.
- Applied text preprocessing, tokenization, and word embeddings.
- Compared model performance using accuracy, precision, recall, and F1-score.
- Analyzed classification behavior and model limitations.

## Technologies

**Python | TensorFlow | Keras | Scikit-learn | Pandas | NumPy | NLP**

## Dataset

The project uses a Turkish phishing email dataset containing **7,504 records**.

| Category | Emails |
|---|---:|
| Phishing | 6,004 |
| Legitimate | 1,500 |

Email subject, sender, and body are combined for binary classification.

## Model Comparison

Eight trained model versions were evaluated using the existing dataset.

| Model | Accuracy | Phishing F1-score |
|---|---:|---:|
| LSTM v1 | 80.01% | 88.90% |
| LSTM v2 | 99.93% | 99.96% |
| BiLSTM v1 | 100.00% | 100.00% |
| BiLSTM v2 | 99.93% | 99.96% |
| CNN-LSTM v1 | 78.88% | 88.19% |
| CNN-LSTM v2 | 100.00% | 100.00% |
| GRU v1 | 78.88% | 88.19% |
| GRU v2 | 78.88% | 88.19% |

### Key Findings

- **BiLSTM v1 and CNN-LSTM v2** achieved the highest scores in this evaluation.
- Several baseline models predicted all emails as phishing, highlighting the importance of class-specific metrics.
- Different architectures and training configurations produced significantly different classification behaviors.

**Evaluation Note:** These results were obtained from previously trained models using the existing dataset. Duplicate records, preprocessing leakage, and differences in evaluation splits may affect the reported scores. They should not be interpreted as independent benchmark results.

## Project Structure

```text
Cybersecurity_projects/
├── data/                 # Turkish phishing dataset
├── models/
│   ├── backup/           # Saved models and tokenizers
│   └── *.py              # Model implementations
├── scripts/              # Training and prediction scripts
└── README.md
```

## Getting Started

Clone the repository:

```bash
git clone https://github.com/sudeilhn/Cybersecurity_projects.git
cd Cybersecurity_projects
```

Install the required libraries:

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn
```

Run a training script, for example:

```bash
python scripts/train_gru.py
```

**Note:** Some scripts contain absolute Windows file paths that must be updated before running the project on another computer.

## Future Improvements

- Standardize training and evaluation splits.
- Remove duplicate records and prevent data leakage.
- Improve legitimate email detection and reduce false positives.
- Evaluate model generalization on unseen datasets.

---
