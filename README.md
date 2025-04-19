# Reddit Post Popularity Classification

## Project Overview

This repository presents an advanced end-to-end pipeline for **classifying Reddit posts based on their popularity** using state-of-the-art **Natural Language Processing (NLP)** and **deep learning** techniques. The framework is designed to handle textual content from social media and predict post popularity levels with high accuracy by integrating **transformer-based language models**, **convolutional architectures**, and **autoencoder-based representations**.

By utilizing **BERT embeddings** as the semantic foundation for the input text, this system demonstrates how modern language modeling can be adapted for classification tasks where subjective popularity (measured via upvote ratios) becomes the target label.

### Classification Objective

Posts are labeled based on the `upvote_ratio`, a Reddit metric indicating the proportion of upvotes compared to total votes:

- **Low Popularity**: upvote ratio ≤ 0.5  
- **Average Popularity**: 0.5 < upvote ratio ≤ 0.8  
- **High Popularity**: upvote ratio > 0.8

This multi-class classification is valuable for content ranking, moderation prioritization, and community engagement modeling.

![Project Cover](https://github.com/sntk-76/Data-Mining/blob/main/Data%20mining.png)

---

## Technology Stack

### Programming Language and Runtime
- **Python 3.7+**

### Core Libraries
- **Data Manipulation**: `pandas`, `numpy`
- **Visualization**: `matplotlib`, `seaborn`, `plotly`
- **Machine Learning**: `scikit-learn`, `imbalanced-learn`
- **Deep Learning Framework**: `TensorFlow` with `Keras`
- **NLP and Transformers**: `HuggingFace Transformers`, `tokenizers`
- **Model Evaluation**: `scikit-learn.metrics`, `classification_report`, `confusion_matrix`

These technologies work in harmony to ensure modular, scalable, and reproducible experimentation.

---

## System Architecture

### 1. **Data Acquisition and Preprocessing**
- Input data is expected in a structured `.csv` format with relevant metadata and post content.
- Columns used: `subreddit`, `title`, `selftext`, `upvote_ratio`.
- Preprocessing pipeline includes:
  - Null value removal and filtering of incomplete records.
  - Concatenation of `title` and `selftext` fields to capture complete context.
  - Stratified labeling based on upvote ratio thresholds.

### 2. **BERT-based Text Embedding**
- Utilizes **Bidirectional Encoder Representations from Transformers (BERT)** models for feature extraction:
  - **Default Model**: `bert-base-uncased`
  - **Optional**: `TinyBERT`, `RoBERTa`, `DistilBERT`, `ALBERT`
- Tokenization using WordPiece-based tokenizer.
- Outputs used:
  - **CLS Token** (`[CLS]`): Captures summary representation.
  - **Pooled Output**: Represents contextualized semantic embedding of the entire input.
- Feature vectors are extracted and cached for efficiency.

### 3. **Class Imbalance Handling**
Given the real-world skew in popularity data, this project implements robust strategies to mitigate class imbalance:
- **Random Oversampling**: Synthetic duplication of minority class samples.
- **Random Undersampling**: Controlled reduction of the majority class.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic samples in feature space using k-nearest neighbors.

### 4. **Model Architectures**

#### A. **Fully Connected Neural Networks**
- BERT embeddings are passed through multiple dense layers with non-linear activation functions (ReLU).
- Dropout and batch normalization are employed to prevent overfitting.
- Output layer uses softmax activation for multi-class classification.

#### B. **Convolutional Neural Networks (CNNs)**
- Embeddings are reshaped into matrix format and passed through:
  - 1D convolutional filters to capture local n-gram semantics.
  - Max-pooling layers for dimensionality reduction and positional invariance.
  - Final dense layers for classification.

#### C. **Autoencoder-Augmented Networks**
- Two-stage process:
  - **Stage 1**: Train an unsupervised autoencoder to compress high-dimensional embeddings into latent space.
  - **Stage 2**: Use the encoder's output as input to a classification head (dense or CNN-based).
- Helps in learning compact, noise-reduced representations of semantic content.

---

## Evaluation Pipeline

### Metrics Tracked:
- **Accuracy**: Overall prediction correctness.
- **Precision, Recall, F1-Score**: Per-class and macro-averaged performance.
- **Confusion Matrices**: For detailed error analysis.
- **Support**: Number of samples per class to contextualize scores.

### Visualization Techniques:
- Comparative plots using `seaborn` and `matplotlib`.
- Precision-recall trade-offs for each class.
- Heatmaps for model confusion matrices.
- Overall model benchmarking on a unified axis.

---

## Key Python Classes and Methods

| Class Name                    | Purpose                                                                 |
|------------------------------|-------------------------------------------------------------------------|
| `LabelClassification`        | Transforms continuous upvote ratios into categorical labels.           |
| `Preprocessing`              | Handles cleaning, embedding, and balancing.                            |
| `NeuralNetwork`              | Base classifier model using dense architecture.                        |
| `ConvolutionalDenseNetwork`  | Hybrid model combining CNN layers with dense outputs.                  |
| `AutoencoderClassifier`      | Compresses semantic representations and classifies latent vectors.     |
| `Visualization`              | Provides statistical visualizations and comparisons.                   |

---

## Results & Analysis

Visual output is saved to the `visualizations/` directory, including:

- `confusion_matrices.png`: Class-wise misclassifications.
- `overall_accuracy.png`: Total accuracy across models.
- `label_accuracy.png`: Accuracy distribution per popularity label.
- `f1_score_comparison.png`: F1 scores per model architecture.
- `precision_recall_per_label.png`: Label-specific trade-offs.
- `confusion_matrix_difference.png`: Delta heatmaps for model comparisons.

---

## Project Highlights

- **Transformer Integration**: Exploits pretrained BERT and its variants for text understanding.
- **Multi-architecture Benchmarking**: Dense, CNN, and hybrid approaches evaluated in parallel.
- **Imbalance Mitigation**: Integrates SMOTE and resampling techniques for realistic datasets.
- **Flexible Pipeline**: Modular components enable adaptation for sentiment analysis, fake news detection, or toxicity classification.

---

## Future Enhancements

- Expand to **multi-modal analysis** by integrating post metadata or image content.
- Incorporate **attention-based pooling mechanisms** for richer sentence representations.
- Implement **cross-validation** and **hyperparameter optimization** using `Optuna` or `Ray Tune`.
- Extend to **zero-shot classification** using `sentence-transformers` and `T5`.

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for usage rights and limitations.
