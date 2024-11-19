## Overview
This project demonstrates a complete pipeline for classification tasks using BERT-based models, CNNs, and autoencoders. The goal is to classify Reddit posts based on their popularity into three categories: "Less Popularity," "Average Popularity," and "Most Popularity." 

The workflow includes:
1. Data preparation and labeling
2. BERT-based embedding extraction
3. Building, training, and evaluating neural network models
4. Visualizing results

---

## Prerequisites

### Libraries and Frameworks:
- Python 3.7+
- Pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- TensorFlow
- Transformers (HuggingFace)
- imbalanced-learn

### Data:
- Input CSV file: `/kaggle/input/ukraine-war/original_data.csv`
  - Required columns: `subreddit`, `title`, `selftext`, `upvote_ratio`

---

## Project Structure

### 1. **Data Preparation**
- Data is loaded and preprocessed to include:
  - Text (`selftext`) for input.
  - Labels based on `upvote_ratio`:
    - ≤ 0.5: "Less Popularity"
    - 0.5 < x ≤ 0.8: "Average Popularity"
    - > 0.8: "Most Popularity"

### 2. **BERT Tokenization and Embedding**
- Pretrained BERT models are used to tokenize the input text and extract embeddings (`pooled output` and `CLS output`).
- Supported models:
  - `bert-base-uncased`
  - (Optional) `TinyBERT`, `RoBERTa`, `ALBERT`

### 3. **Data Balancing**
- Balancing techniques:
  - Oversampling
  - Undersampling
  - SMOTE augmentation

### 4. **Model Architectures**
- **Dense Neural Networks**:
  - Takes BERT embeddings as input.
- **Convolutional Neural Networks (CNNs)**:
  - Applies 1D convolutional layers to embeddings.
- **Autoencoders**:
  - Learns compressed representations of embeddings before classification.

### 5. **Evaluation and Visualization**
- Models are evaluated using:
  - Confusion matrices
  - Classification reports
- Results visualization:
  - Confusion matrices
  - Overall accuracy comparison
  - Label-specific accuracy
  - Precision, recall, and F1-score

---

## Instructions

### 1. Clone the Repository:
```bash
git clone <https://github.com/sntk-76/Data-Mining>
cd <repository-folder>
```

### 2. Install Dependencies:
```bash
pip install -r requirements.txt
```

### 3. Prepare the Dataset:
- Place the dataset CSV in the appropriate folder.
- Update the file path if necessary.

### 4. Run the Pipeline:
Run the main script in order, or execute all cells in the provided Jupyter Notebook.

---

## Key Classes and Methods

### 1. **Label_classification**
- Labels the data based on the `upvote_ratio`.

### 2. **preprocessing**
- Handles tokenization, train-test splitting, tensor conversion, and data augmentation.

### 3. **neural_network**
- Base class for training and evaluating dense neural networks.

### 4. **ConvolutionalDenseNetwork**
- Extends `neural_network` with 1D convolutional layers.

### 5. **AutoencoderClassifierNetwork**
- Extends `neural_network` with autoencoder-based compression.

### 6. **Visualization**
- Methods for visualizing confusion matrices, accuracy, F1 scores, and more.

---

## Results
- Final evaluation results include confusion matrices and accuracy comparisons for all models.
- Visualizations are saved as PNG files:
  - `confusion_matrices.png`
  - `overall_accuracy.png`
  - `label_accuracy.png`
  - `f1_score_comparison.png`
  - `precision_recall_per_label.png`
  - `confusion_matrix_difference.png`

---

## Future Work
- Experiment with additional BERT variants (e.g., TinyBERT, RoBERTa, ALBERT).
- Optimize hyperparameters for better performance.
- Extend the visualization module to include more metrics.

---

## License
This project is licensed under the MIT License.
