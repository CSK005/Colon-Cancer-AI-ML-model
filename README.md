Colon Cancer Classifier Models Using Exome Sequencing Data

This repository contains a Streamlit-based web application for the classification of colon cancer using annotated exome sequencing data. The application enables users to upload variant call CSV files, preprocess the data, train classification models, evaluate performance, and compare model accuracy. The tool is intended to support researchers and clinicians in exploring the utility of machine learning and deep learning models for early detection of colon cancer.

## Overview

This application performs the following tasks:

* Data preprocessing (cleaning, label encoding, normalization)
* Model training using three different classifiers:

  * Random Forest (scikit-learn)
  * XGBoost (XGBoost library)
  * Deep Neural Network (TensorFlow/Keras)
* Evaluation using accuracy, confusion matrices, and classification metrics
* Interactive visualization of feature distributions
* Summary report of model performance

## Installation

1. Clone the repository:

```bash
git clone https://github.com/CSK005/Colon-Cancer-AI-ML-model.git
cd Colon-Cancer-AI-ML-model
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is not available, install dependencies manually:

```bash
pip install streamlit pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
```

## Running the Application

To launch the Streamlit app:

```bash
streamlit run app.py
```

This will open the application in a web browser at `http://localhost:8501`.

## Input Format

The application accepts `.csv` files containing annotated exome data. Input files must include the following fields (column names should match exactly):

* **Genomic coordinates**: `Chr`, `Start`, `End`
* **Functional annotations**: `Func.refGene`, `ExonicFunc.refGene`
* **Predictive scores and categories**: `SIFT_pred`, `Polyphen2_HDIV_pred`, `Polyphen2_HVAR_pred`, `MutationTaster_pred`, `MutationAssessor_pred`, `CLNSIG`, `CADD`, `CADD_Phred`, `AF`, `AF_popmax`, etc.

The application automatically handles:

* Missing value imputation
* Label encoding of categorical features
* Scaling of numerical features

A preview of the expected data format is shown within the app interface.

## Models and Evaluation

### 1. Random Forest

* Implementation: scikit-learn
* Metrics: accuracy, confusion matrix, classification report

### 2. XGBoost

* Implementation: XGBoost Python API
* Custom parameters: `eval_metric='mlogloss'`, `use_label_encoder=False`
* Metrics: same as above

### 3. Deep Neural Network (DNN)

* Implementation: TensorFlow/Keras Sequential model
* Architecture: 3 hidden layers with ReLU activation, softmax output
* Loss: Sparse categorical crossentropy
* Optimizer: Adam
* Evaluation: accuracy, confusion matrix, classification report

The application presents visual outputs for each model, including:

* Confusion matrices
* Bar plots of precision, recall, and F1-score for each class
* Final comparison table of accuracies

## Application Usage

1. Upload one or more CSV files for training data.
2. Upload one or more CSV files for testing data.
3. The app will:

   * Display dataset previews
   * Visualize selected feature distributions
   * Train and evaluate all three models
   * Present confusion matrices and performance metrics
   * Summarize accuracy across all models

## Developer Team

This application was developed by:

* Chandrashekar K
* Dr. Vidya Niranjan
* Anagha S Setlur

Department of Biotechnology, RV College of Engineering, Bengaluru, India

## Citation and Acknowledgment

If this tool contributes to your research, please cite the GitHub repository and acknowledge the development team.

## Repository Link

Codebase: [https://github.com/CSK005/Colon-Cancer-AI-ML-model](https://github.com/CSK005/Colon-Cancer-AI-ML-model)

---

Let me know if you would like to include license information, example screenshots, or a formal citation section for manuscript referencing.
