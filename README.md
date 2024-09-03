# Web Application Attack Detection Using Machine Learning Models

## Introduction

Web application security is a critical concern in today's digital age, where attacks on web applications are increasingly frequent and sophisticated. Despite the abundance of web traffic data, publicly available datasets for training machine learning models to detect such attacks are relatively rare. This project leverages the CSIC 2010 Dataset, a well-known dataset in the field, to develop and compare several machine learning models for the detection of web application attacks.

## Dataset Overview

### Source
The dataset used in this project is the **CSIC 2010 Dataset**, a comprehensive collection of HTTP request logs, including both normal and malicious traffic. The dataset was designed for web intrusion detection research and includes a variety of attack types, such as SQL injection, buffer overflow, and directory traversal.

### Dataset Details
- **Total Records:** 61,065
- **Columns:** 17
  - **Method:** The HTTP request method used (e.g., GET, POST).
  - **User-Agent:** Details about the client making the request.
  - **Pragma & Cache-Control:** Caching directives.
  - **Accept, Accept-Encoding, Accept-Charset:** Content types, encodings, and character sets accepted by the client.
  - **Language:** Language preferences.
  - **Host:** The server's hostname.
  - **Cookie:** Cookies sent with the request.
  - **Content-Type:** Media type of the request body.
  - **Connection:** Indicates whether the connection should remain open.
  - **Length & Content:** Length and content of the request or response body.
  - **Classification:** Indicates whether the request is normal or anomalous.
  - **URL:** The URL requested.

### Data Preprocessing
Given the nature of the dataset, significant preprocessing was required, particularly for the URL field, which was carefully parsed and tokenized to ensure that relevant features could be extracted for model training.

## Machine Learning Models

The following machine learning models were developed and evaluated for detecting web application attacks:

1. **Random Forest**
   - An ensemble learning method that builds multiple decision trees and merges them to get a more accurate and stable prediction.

2. **K-Nearest Neighbors (KNN)**
   - A simple, instance-based learning algorithm that classifies data points based on the majority class of their nearest neighbors.

3. **Decision Tree**
   - A model that makes decisions based on the feature values of the dataset, organizing them into a tree structure.

4. **Gradient Descent**
   - An optimization algorithm used to minimize the loss function in models by iteratively adjusting the parameters.

5. **Artificial Neural Network (ANN)**
   - A deep learning model that mimics the way human brains process information, consisting of multiple layers of interconnected neurons.

6. **Multi-Layer Perceptron Classifier (MLPC)**
   - A type of ANN specifically designed for classification tasks, capable of capturing complex patterns in the data.

## Project Workflow

1. **Data Preprocessing:**
   - URL processing: Parsing and tokenization of the URL field.
   - Feature encoding: Handling categorical data through one-hot encoding and label encoding.
   - Data normalization: Scaling features to a uniform range.

2. **Exploratory Data Analysis (EDA):**
   - Visualization of feature distributions.
   - Correlation analysis to identify significant relationships between features and the target variable.

3. **Model Training & Evaluation:**
   - Models were trained on the processed dataset using cross-validation techniques.
   - Performance metrics, including accuracy, precision, recall, F1-score, and ROC-AUC, were used to evaluate each model's effectiveness.

4. **Model Comparison:**
   - A comprehensive comparison of the models based on their performance metrics.
   - Insights into which models perform best in detecting different types of attacks.

## Results

- **Random Forest:** Achieved high accuracy with balanced performance across all classes.
- **KNN:** Provided good results but was sensitive to the choice of k and computationally expensive.
- **Decision Tree:** Simple and interpretable, with decent accuracy but prone to overfitting.
- **Gradient Descent:** Effective optimization, especially when combined with other models.
- **ANN & MLPC:** Demonstrated the potential of deep learning models to capture complex patterns, outperforming classical models in many scenarios.

## Conclusion

The project successfully demonstrates the application of various machine learning models to the detection of web application attacks. While traditional models like Random Forest and Decision Tree provide solid baselines, deep learning approaches such as ANN and MLPC offer enhanced performance, especially in handling complex, high-dimensional data like HTTP logs.

## Future Work

- **Hyperparameter Tuning:** Further optimization of models through more extensive hyperparameter tuning.
- **Ensemble Methods:** Combining multiple models to create an even more robust detection system.
- **Real-Time Deployment:** Implementing the best-performing model in a real-time web security environment for continuous monitoring and threat detection.

## Getting Started

### Prerequisites
- Python 3.x
- Jupyter Notebook
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `tensorflow`, `keras`, `matplotlib`, `seaborn`

### Installation

Clone the repository:
```bash
git clone https://github.com/yourusername/web-attack-detection-ml.git