# Fraud Detection using AutoEncoder (PyOD)

## Project Overview

This project implements an unsupervised deep learning approach for detecting fraudulent credit card transactions. An AutoEncoder model from the PyOD library is used to identify anomalies by learning the normal behavior of transactions and detecting deviations.

## Objective

The objective of this project is to build a fraud detection system using an AutoEncoder model that can effectively identify anomalous (fraudulent) transactions in a highly imbalanced dataset.

## Dataset

* **Source:** Kaggle Credit Card Fraud Detection Dataset
* The dataset contains anonymized transaction features
* **Target variable:**

  * `0`: Normal transaction
  * `1`: Fraudulent transaction

## Methodology

### 1. Data Loading

The dataset is loaded using pandas for further processing.

### 2. Data Preprocessing

* Features and target variable are separated
* Data is normalized using StandardScaler

### 3. Model Building

* An AutoEncoder model from PyOD is used
* The model learns compressed representations of normal transactions

### 4. Model Training

* The model is trained on the entire dataset (unsupervised learning)

### 5. Prediction

* The model predicts anomalies based on reconstruction error
* Output:

  * `0`: Normal
  * `1`: Fraud

### 6. Evaluation

* Classification Report (precision, recall, F1-score)
* Confusion Matrix
* Visualization of predicted results

## Technologies Used

* Python
* PyOD
* PyTorch
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Google Colab

## Installation

Install required libraries using:

```bash
pip install pyod torch pandas numpy scikit-learn matplotlib seaborn
```

## How to Run

1. Open the notebook in Google Colab
2. Upload the dataset file (`creditcard.csv`)
3. Install dependencies
4. Run all cells

## Results

* **Accuracy:** Approximately 98%
* **Fraud Recall:** High (model detects majority of fraud cases)
* Precision for fraud class is low due to class imbalance
* The model effectively identifies anomalous transactions

## Output

Include:

* Classification Report
* Confusion Matrix
* Visualization graph

## Repository Structure

```
.
├── fraud_detection.ipynb
├── requirements.txt
├── README.md
└── screenshots/
```

## Conclusion

The AutoEncoder model successfully detects fraudulent transactions by identifying anomalies in the dataset. This demonstrates the effectiveness of unsupervised deep learning techniques in fraud detection systems, especially in scenarios with highly imbalanced data.

## Future Improvements

* Hyperparameter tuning of AutoEncoder
* Comparison with other anomaly detection models
* Use of ensemble techniques
* Real-time fraud detection pipeline
