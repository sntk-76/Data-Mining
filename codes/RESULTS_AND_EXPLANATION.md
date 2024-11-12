# Summary of the results

We will analyze the following models:

- BERT base: Original model
- TinyBERT: Further reduced for speed, but with lower accuracy
- RoBERTa: BERT-like but trained on larger datasets with different preprocessing
- ALBERT: designed to be lighter and faster than BERT

And we'll feed them to the following models:
- Feedforward Neural Network
- Progressive Neural Network
- Convolutional Neural Network
- Autoencoder Neural Network
- Attention-based Neural Network

Please note that **pooler embeddings** aggregate token information into a single vector, ideal for generalized feature extraction. **CLS embeddings** instead focus on specific token representations, generally leading to nuanced distinctions that may slightly improve class separation.

## Feedforward Neural Network
The Feedforward Neural Network utilizes fully connected layers to process embeddings for classification tasks, relying on dense layers to capture features in the Pooler or CLS representations. Different models are presented.

### BERT base

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.80      | 0.72   | 0.75     | 141     |
| 1         | 0.92      | 0.95   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.90      | 0.87   | 0.88     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.91     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.85      | 0.67   | 0.75     | 141     |
| 1         | 0.90      | 0.97   | 0.93     | 575     |
| 2         | 0.99      | 0.88   | 0.93     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.91      | 0.84   | 0.87     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.90     | 828     |

### Albert

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.88      | 0.66   | 0.75     | 141     |
| 1         | 0.91      | 0.98   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.92**    |
| **macro avg**   | 0.93      | 0.86   | 0.89     | 828     |
| **weighted avg**| 0.92      | 0.92   | 0.91     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.88      | 0.66   | 0.75     | 141     |
| 1         | 0.91      | 0.98   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.92**    |
| **macro avg**   | 0.93      | 0.86   | 0.89     | 828     |
| **weighted avg**| 0.92      | 0.92   | 0.91     | 828     |

### Roberta

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.77      | 0.73   | 0.75     | 141     |
| 1         | 0.92      | 0.94   | 0.93     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.89      | 0.87   | 0.88     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.91     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.83      | 0.70   | 0.76     | 141     |
| 1         | 0.92      | 0.96   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.91      | 0.87   | 0.89     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.91     | 828     |

### Tinybert

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.81      | 0.74   | 0.77     | 141     |
| 1         | 0.93      | 0.96   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.92**    |
| **macro avg**   | 0.91      | 0.88   | 0.89     | 828     |
| **weighted avg**| 0.92      | 0.92   | 0.92     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.76      | 0.79   | 0.77     | 141     |
| 1         | 0.94      | 0.94   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.89      | 0.89   | 0.89     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.91     | 828     |


## Residual Neural Network
Residual Neural Networks introduce skip connections, allowing the model to retain key information from earlier layers. It enables better feature retention across layers.

### BERT base

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.69      | 0.84   | 0.75     | 141     |
| 1         | 0.95      | 0.90   | 0.92     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.90**    |
| **macro avg**   | 0.87      | 0.89   | 0.88     | 828     |
| **weighted avg**| 0.91      | 0.90   | 0.90     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.77      | 0.74   | 0.76     | 141     |
| 1         | 0.93      | 0.94   | 0.93     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.89      | 0.87   | 0.88     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.91     | 828     |

### Albert

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.90      | 0.63   | 0.74     | 141     |
| 1         | 0.91      | 0.98   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.92**    |
| **macro avg**   | 0.93      | 0.85   | 0.88     | 828     |
| **weighted avg**| 0.92      | 0.92   | 0.91     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.81      | 0.74   | 0.77     | 141     |
| 1         | 0.93      | 0.96   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.92**    |
| **macro avg**   | 0.91      | 0.88   | 0.89     | 828     |
| **weighted avg**| 0.92      | 0.92   | 0.92     | 828     |

### Roberta

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.44      | 0.91   | 0.59     | 141     |
| 1         | 0.97      | 0.70   | 0.81     | 575     |
| 2         | 0.92      | 1.00   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.77**    |
| **macro avg**   | 0.78      | 0.87   | 0.79     | 828     |
| **weighted avg**| 0.87      | 0.77   | 0.79     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.73      | 0.74   | 0.74     | 141     |
| 1         | 0.93      | 0.93   | 0.93     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.90**    |
| **macro avg**   | 0.88      | 0.87   | 0.88     | 828     |
| **weighted avg**| 0.90      | 0.90   | 0.90     | 828     |

### Tinybert

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 1.00      | 0.47   | 0.64     | 141     |
| 1         | 0.88      | 1.00   | 0.93     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.90**    |
| **macro avg**   | 0.96      | 0.80   | 0.84     | 828     |
| **weighted avg**| 0.91      | 0.90   | 0.89     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.83      | 0.71   | 0.76     | 141     |
| 1         | 0.92      | 0.96   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.92**    |
| **macro avg**   | 0.91      | 0.87   | 0.89     | 828     |
| **weighted avg**| 0.91      | 0.92   | 0.91     | 828     |

## Progressive Neural Network
Progressive Neural Networks leverage lateral connections to enable knowledge sharing across tasks or layers.

### BERT base

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.36      | 0.79   | 0.50     | 141     |
| 1         | 0.92      | 0.65   | 0.76     | 575     |
| 2         | 0.93      | 0.97   | 0.95     | 112     |
| **accuracy**    |           |        |          | **0.72**    |
| **macro avg**   | 0.74      | 0.80   | 0.74     | 828     |
| **weighted avg**| 0.83      | 0.72   | 0.74     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.85      | 0.67   | 0.75     | 141     |
| 1         | 0.91      | 0.97   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.92      | 0.86   | 0.88     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.91     | 828     |

### Albert

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.67      | 0.13   | 0.21     | 141     |
| 1         | 0.79      | 0.99   | 0.88     | 575     |
| 2         | 0.99      | 0.71   | 0.82     | 112     |
| **accuracy**    |           |        |          | **0.81**    |
| **macro avg**   | 0.81      | 0.61   | 0.64     | 828     |
| **weighted avg**| 0.80      | 0.81   | 0.76     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.90      | 0.63   | 0.74     | 141     |
| 1         | 0.91      | 0.98   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.92**    |
| **macro avg**   | 0.93      | 0.85   | 0.88     | 828     |
| **weighted avg**| 0.92      | 0.92   | 0.91     | 828     |

### Roberta

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.86      | 0.17   | 0.28     | 141     |
| 1         | 0.71      | 0.99   | 0.83     | 575     |
| 2         | 0.00      | 0.00   | 0.00     | 112     |
| **accuracy**    |           |        |          | **0.72**    |
| **macro avg**   | 0.52      | 0.39   | 0.37     | 828     |
| **weighted avg**| 0.64      | 0.72   | 0.63     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.81      | 0.74   | 0.77     | 141     |
| 1         | 0.93      | 0.96   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.92**    |
| **macro avg**   | 0.91      | 0.88   | 0.89     | 828     |
| **weighted avg**| 0.92      | 0.92   | 0.92     | 828     |

### Tinybert

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.83      | 0.71   | 0.76     | 141     |
| 1         | 0.92      | 0.96   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.92**    |
| **macro avg**   | 0.91      | 0.87   | 0.89     | 828     |
| **weighted avg**| 0.91      | 0.92   | 0.91     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.89      | 0.66   | 0.76     | 141     |
| 1         | 0.91      | 0.98   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.92**    |
| **macro avg**   | 0.93      | 0.86   | 0.89     | 828     |
| **weighted avg**| 0.92      | 0.92   | 0.92     | 828     |


## Convolutional Neural Network
Convolutional Neural Networks (CNNs) apply convolutional filters to capture spatial or sequential patterns, often useful in distinguishing subtle class differences.

### BERT base

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.76      | 0.68   | 0.72     | 141     |
| 1         | 0.91      | 0.95   | 0.93     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.90**    |
| **macro avg**   | 0.89      | 0.85   | 0.87     | 828     |
| **weighted avg**| 0.90      | 0.90   | 0.90     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.84      | 0.67   | 0.75     | 141     |
| 1         | 0.91      | 0.97   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.91      | 0.86   | 0.88     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.91     | 828     |

### Albert

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.76      | 0.75   | 0.76     | 141     |
| 1         | 0.93      | 0.93   | 0.93     | 575     |
| 2         | 0.93      | 0.97   | 0.95     | 112     |
| **accuracy**    |           |        |          | **0.90**    |
| **macro avg**   | 0.88      | 0.88   | 0.88     | 828     |
| **weighted avg**| 0.90      | 0.90   | 0.90     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.87      | 0.69   | 0.77     | 141     |
| 1         | 0.92      | 0.96   | 0.94     | 575     |
| 2         | 0.93      | 0.97   | 0.95     | 112     |
| **accuracy**    |           |        |          | **0.92**    |
| **macro avg**   | 0.91      | 0.87   | 0.89     | 828     |
| **weighted avg**| 0.91      | 0.92   | 0.91     | 828     |

### Roberta

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.87      | 0.60   | 0.71     | 141     |
| 1         | 0.90      | 0.98   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.92      | 0.84   | 0.87     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.90     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.73      | 0.74   | 0.74     | 141     |
| 1         | 0.93      | 0.93   | 0.93     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.90**    |
| **macro avg**   | 0.88      | 0.87   | 0.88     | 828     |
| **weighted avg**| 0.90      | 0.90   | 0.90     | 828     |

### Tinybert

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.85      | 0.62   | 0.72     | 141     |
| 1         | 0.90      | 0.97   | 0.94     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.92      | 0.84   | 0.87     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.90     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.72      | 0.82   | 0.76     | 141     |
| 1         | 0.93      | 0.92   | 0.93     | 575     |
| 2         | 0.99      | 0.88   | 0.93     | 112     |
| **accuracy**    |           |        |          | **0.90**    |
| **macro avg**   | 0.88      | 0.87   | 0.87     | 828     |
| **weighted avg**| 0.90      | 0.90   | 0.90     | 828     |

## Autoencoder Neural Network
Autoencoders are designed for dimensionality reduction and reconstruction. The CLS embedding can sometimes retain classification-specific features better than Pooler when using an autoencoder.

### BERT base

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.86      | 0.45   | 0.60     | 141     |
| 1         | 0.87      | 0.98   | 0.92     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.89**    |
| **macro avg**   | 0.91      | 0.79   | 0.83     | 828     |
| **weighted avg**| 0.89      | 0.89   | 0.87     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.84      | 0.60   | 0.70     | 141     |
| 1         | 0.90      | 0.97   | 0.93     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.90**    |
| **macro avg**   | 0.91      | 0.84   | 0.87     | 828     |
| **weighted avg**| 0.90      | 0.90   | 0.90     | 828     |

### Albert

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.74      | 0.48   | 0.58     | 141     |
| 1         | 0.87      | 0.94   | 0.91     | 575     |
| 2         | 0.93      | 0.92   | 0.92     | 112     |
| **accuracy**    |           |        |          | **0.86**    |
| **macro avg**   | 0.85      | 0.78   | 0.80     | 828     |
| **weighted avg**| 0.85      | 0.86   | 0.85     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.83      | 0.71   | 0.76     | 141     |
| 1         | 0.92      | 0.96   | 0.94     | 575     |
| 2         | 1.00      | 0.91   | 0.95     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.91      | 0.86   | 0.89     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.91     | 828     |

### Roberta

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.00      | 0.00   | 0.00     | 141     |
| 1         | 0.69      | 1.00   | 0.82     | 575     |
| 2         | 0.00      | 0.00   | 0.00     | 112     |
| **accuracy**    |           |        |          | **0.69**    |
| **macro avg**   | 0.23      | 0.33   | 0.27     | 828     |
| **weighted avg**| 0.48      | 0.69   | 0.57     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.36      | 0.63   | 0.46     | 141     |
| 1         | 0.81      | 0.79   | 0.80     | 575     |
| 2         | 1.00      | 0.21   | 0.35     | 112     |
| **accuracy**    |           |        |          | **0.68**    |
| **macro avg**   | 0.72      | 0.54   | 0.54     | 828     |
| **weighted avg**| 0.76      | 0.68   | 0.68     | 828     |

### Tinybert

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.00      | 0.00   | 0.00     | 141     |
| 1         | 0.69      | 1.00   | 0.82     | 575     |
| 2         | 0.00      | 0.00   | 0.00     | 112     |
| **accuracy**    |           |        |          | **0.69**    |
| **macro avg**   | 0.23      | 0.33   | 0.27     | 828     |
| **weighted avg**| 0.48      | 0.69   | 0.57     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.80      | 0.66   | 0.72     | 141     |
| 1         | 0.91      | 0.96   | 0.93     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.90**    |
| **macro avg**   | 0.90      | 0.85   | 0.87     | 828     |
| **weighted avg**| 0.90      | 0.90   | 0.90     | 828     |


## Attention-based Neural Network
Attention-based models focus on significant parts of the input, where particular tokens influence the output more than others. CLS tokens benefit from attention by maintaining focused, class-distinguishing information.

### BERT base

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.74      | 0.71   | 0.72     | 141     |
| 1         | 0.88      | 0.94   | 0.91     | 575     |
| 2         | 1.00      | 0.71   | 0.83     | 112     |
| **accuracy**    |           |        |          | **0.87**    |
| **macro avg**   | 0.87      | 0.78   | 0.82     | 828     |
| **weighted avg**| 0.87      | 0.87   | 0.87     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.83      | 0.71   | 0.76     | 141     |
| 1         | 0.92      | 0.96   | 0.94     | 575     |
| 2         | 1.00      | 0.91   | 0.95     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.91      | 0.86   | 0.89     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.91     | 828     |

### Albert

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.47      | 0.93   | 0.62     | 141     |
| 1         | 0.97      | 0.73   | 0.83     | 575     |
| 2         | 0.93      | 0.97   | 0.95     | 112     |
| **accuracy**    |           |        |          | **0.79**    |
| **macro avg**   | 0.79      | 0.88   | 0.80     | 828     |
| **weighted avg**| 0.88      | 0.79   | 0.81     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.76      | 0.76   | 0.76     | 141     |
| 1         | 0.94      | 0.93   | 0.93     | 575     |
| 2         | 0.93      | 0.97   | 0.95     | 112     |
| **accuracy**    |           |        |          | **0.90**    |
| **macro avg**   | 0.88      | 0.89   | 0.88     | 828     |
| **weighted avg**| 0.90      | 0.90   | 0.90     | 828     |

### Roberta

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.32      | 0.89   | 0.47     | 141     |
| 1         | 0.94      | 0.71   | 0.81     | 575     |
| 2         | 0.00      | 0.00   | 0.00     | 112     |
| **accuracy**    |           |        |          | **0.64**    |
| **macro avg**   | 0.42      | 0.53   | 0.42     | 828     |
| **weighted avg**| 0.71      | 0.64   | 0.64     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.76      | 0.77   | 0.76     | 141     |
| 1         | 0.94      | 0.93   | 0.93     | 575     |
| 2         | 0.93      | 0.97   | 0.95     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.88      | 0.89   | 0.88     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.91     | 828     |

### Tinybert

**Results for pooling layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.95      | 0.42   | 0.58     | 141     |
| 1         | 0.87      | 0.99   | 0.92     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.89**    |
| **macro avg**   | 0.94      | 0.78   | 0.82     | 828     |
| **weighted avg**| 0.90      | 0.89   | 0.87     | 828     |

**Results for CLS layer**

|           | precision | recall | f1-score | support |
|-----------|-----------|--------|----------|---------|
| 0         | 0.76      | 0.75   | 0.76     | 141     |
| 1         | 0.93      | 0.94   | 0.93     | 575     |
| 2         | 0.99      | 0.94   | 0.96     | 112     |
| **accuracy**    |           |        |          | **0.91**    |
| **macro avg**   | 0.89      | 0.88   | 0.88     | 828     |
| **weighted avg**| 0.91      | 0.91   | 0.91     | 828     |
