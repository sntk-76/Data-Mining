# README

## Overview
This project implements a comprehensive pipeline for classifying Reddit posts based on their popularity using advanced Natural Language Processing (NLP) techniques and machine learning models. It leverages BERT-based embeddings, Convolutional Neural Networks (CNNs), and Autoencoder-based architectures to deliver a robust and flexible classification framework. 

The goal is to categorize Reddit posts into three levels of popularity:
- **Less Popularity**: Upvote ratio ≤ 0.5
- **Average Popularity**: 0.5 < Upvote ratio ≤ 0.8
- **Most Popularity**: Upvote ratio > 0.8

The pipeline includes:
1. Data preparation and labeling.
2. Pretrained BERT embedding extraction for text representation.
3. Model training using different neural network architectures.
4. Evaluation of model performance through multiple metrics.
5. Visualization of results for comparative analysis.

This project demonstrates how cutting-edge techniques can address real-world NLP classification challenges, with the flexibility to adapt to various datasets and use cases.

---

## Prerequisites

### Libraries and Frameworks:
Ensure the following libraries are installed:
- **Core Libraries**: `Python 3.7+`, `Pandas`, `NumPy`, `Matplotlib`, `Seaborn`
- **Machine Learning**: `scikit-learn`, `TensorFlow`, `imbalanced-learn`
- **NLP**: `HuggingFace Transformers`

Install all required dependencies:
```bash
pip install -r requirements.txt
```

### Dataset:
Place the input dataset at the specified path: `/kaggle/input/ukraine-war/original_data.csv`.

The dataset must include the following columns:
- **subreddit**: Name of the subreddit.
- **title**: Title of the Reddit post.
- **selftext**: Main text content of the post.
- **upvote_ratio**: Ratio of upvotes to total votes.

---

## Project Structure

### 1. **Data Preparation**
- **Data Cleaning**: Remove rows with missing or invalid data.
- **Label Generation**: Categorize posts into "Less Popularity," "Average Popularity," and "Most Popularity" based on `upvote_ratio`.

### 2. **BERT Tokenization and Embedding**
- Text data is tokenized using pretrained BERT models:
  - **Default**: `bert-base-uncased`
  - **Optional Models**: `TinyBERT`, `RoBERTa`, `ALBERT`
- Extract embeddings:
  - **Pooled Output**: Encodes the entire input text.
  - **CLS Token Output**: Embedding of the `[CLS]` token.

### 3. **Data Balancing**
To handle imbalanced classes, three strategies are available:
- **Oversampling**: Replicate data from underrepresented classes.
- **Undersampling**: Reduce samples from overrepresented classes.
- **SMOTE**: Synthesize new samples for underrepresented classes.

### 4. **Model Architectures**
#### a. **Dense Neural Networks**
- Input: BERT embeddings (pooled or CLS output).
- Fully connected layers with ReLU activation and dropout for regularization.

#### b. **Convolutional Neural Networks (CNNs)**
- Input: Reshaped BERT embeddings.
- Apply 1D convolutional layers for feature extraction.

#### c. **Autoencoder-based Models**
- Stage 1: Train an autoencoder to learn compressed representations.
- Stage 2: Use the compressed representations for classification.

### 5. **Evaluation and Visualization**
- **Evaluation Metrics**:
  - Confusion Matrix
  - Classification Report (Precision, Recall, F1 Score)
- **Visualization**:
  - Confusion matrices for all models.
  - Overall accuracy comparison.
  - Label-specific accuracy and F1 scores.
  - Precision and recall comparisons.

---

## Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/sntk-76/Data-Mining
cd Data-Mining
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset
Ensure the dataset is placed at `/kaggle/input/ukraine-war/original_data.csv`. Update the path in the code if needed.

### 4. Run the Code
- Execute the script sequentially.
- Alternatively, run all cells in the provided Jupyter Notebook.

### 5. View Results
Generated visualizations will be saved in the working directory.

---

## Key Components and Methods

### Classes:
1. **Label_classification**:
   - Classifies posts based on their `upvote_ratio`.

2. **preprocessing**:
   - Handles data tokenization, train-test splitting, and balancing.

3. **neural_network**:
   - Base class for Dense Neural Networks.

4. **ConvolutionalDenseNetwork**:
   - Extends `neural_network` to include 1D convolutional layers.

5. **AutoencoderClassifierNetwork**:
   - Extends `neural_network` with an autoencoder stage.

6. **Visualization**:
   - Visualizes results using confusion matrices, accuracy metrics, and F1 scores.

---

## Results
The project outputs the following visualizations for easy analysis:
1. **Confusion Matrices**: `confusion_matrices.png`
2. **Overall Accuracy**: `overall_accuracy.png`
3. **Label-specific Accuracy**: `label_accuracy.png`
4. **F1 Score Comparison**: `f1_score_comparison.png`
5. **Precision and Recall**: `precision_recall_per_label.png`
6. **Confusion Matrix Differences**: `confusion_matrix_difference.png`

---

## Highlights
- **Scalable Framework**: Easily extendable for new datasets or BERT variants.
- **Comprehensive Models**: Combines dense, convolutional, and autoencoder architectures.
- **Visual Insights**: Graphical representations for enhanced interpretability.
- **Balanced Data Handling**: Offers multiple techniques to manage class imbalance.

---

## Future Work
1. **Explore Additional Models**:
   - Add more BERT variants like `TinyBERT`, `RoBERTa`, and `ALBERT`.
2. **Hyperparameter Tuning**:
   - Optimize learning rates, layer sizes, and dropout rates.
3. **Advanced Visualization**:
   - Incorporate new metrics like ROC curves and feature importance.
4. **Cross-domain Adaptation**:
   - Test on datasets from other domains to assess generalizability.

---

## License
This project is licensed under the MIT License. See the LICENSE file for more details.
```

This extended README provides detailed explanations, emphasizes project flexibility, and highlights its strengths. The added sections for highlights, scalability, and future work make it more comprehensive.
