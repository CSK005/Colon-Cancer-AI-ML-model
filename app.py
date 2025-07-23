# colon_cancer_app.py

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import io

st.set_page_config(page_title="Colon Cancer Classifier", layout="wide")
st.title("Colon Cancer Classifiers")
st.markdown("""
This application predicts the functional impact of exome variants related to colorectal cancer using three machine learning models:
- Random Forest (RF)
- XGBoost (XGB)
- Deep Neural Network (DNN)

**Instructions:** Upload two CSV files (training and testing datasets) with annotated variant information. The required target column is `Func.refGene`.
""")

# Sample Dataset Format
with st.expander("Click to see sample dataset format"):
    st.markdown("Required columns:")
    example_data = {
        'Chr': ['chr1', 'chr2'],
        'Start': [12345, 67890],
        'End': [12355, 67900],
        'Func.refGene': ['exonic', 'intronic'],
        'ExonicFunc.refGene': ['nonsynonymous', 'synonymous'],
        'Polyphen2_HDIV_pred': ['benign', 'possibly_damaging'],
        'Polyphen2_HVAR_pred': ['benign', 'benign'],
        'SIFT_pred': ['tolerated', 'deleterious'],
        'MutationTaster_pred': ['disease_causing', 'polymorphism'],
        'MutationAssessor_pred': ['low', 'medium'],
        'CLNSIG': ['pathogenic', 'benign'],
        'CADD': [12.3, 23.5],
        'CADD_Phred': [20.1, 25.3],
        'MutationTaster_score': [0.8, 0.2],
        'MutationAssessor_score': [1.2, 2.3],
        'AF': [0.01, 0.05],
        'AF_popmax': [0.02, 0.06]
    }
    st.dataframe(pd.DataFrame(example_data))

# Sidebar for upload
st.sidebar.header("Upload your files")
train_files = st.sidebar.file_uploader("Training Data CSV", type=["csv"], accept_multiple_files=True)
test_files = st.sidebar.file_uploader("Testing Data CSV", type=["csv"], accept_multiple_files=True)

@st.cache_data
def preprocess(files):
    if not files:
        return None, None
    try:
        dfs = [pd.read_csv(file) for file in files]
        df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"Error reading files: {e}")
        return None, None

    df.replace('.', np.nan, inplace=True)
    df.fillna(0, inplace=True)

    chr_map = {f'chr{i}': i for i in range(1, 23)}
    chr_map.update({'chrX': 23, 'chrY': 24})
    df['Chr'] = df['Chr'].map(chr_map).fillna(0).astype(int)

    encode_cols = ["Func.refGene", "ExonicFunc.refGene", "Polyphen2_HDIV_pred", "Polyphen2_HVAR_pred",
                   "SIFT_pred", "MutationTaster_pred", "MutationAssessor_pred", "CLNSIG"]
    for col in encode_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    for col in df.select_dtypes(include=['object']).columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    scale_cols = ["CADD", "CADD_Phred", "MutationTaster_score", "MutationAssessor_score", "AF", "AF_popmax"]
    scale_cols = [col for col in scale_cols if col in df.columns]
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    if 'Func.refGene' not in df.columns:
        st.error("Missing target column 'Func.refGene'")
        return None, None

    X = df.drop(columns=['Func.refGene'])
    y = df['Func.refGene']
    return X, y

if train_files and test_files:
    X_train, y_train = preprocess(train_files)
    X_test, y_test = preprocess(test_files)

    st.subheader("Model Training and Evaluation")
    models_to_run = st.multiselect("Choose model(s) to run", ["Random Forest", "XGBoost", "DNN"], default=["Random Forest"])

    results = {}

    def evaluate(model, X_test, y_test):
        pred = model.predict(X_test)
        acc = accuracy_score(y_test, pred)
        report = classification_report(y_test, pred, output_dict=True)
        cm = confusion_matrix(y_test, pred)
        return acc, report, cm

    if "Random Forest" in models_to_run:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        acc, rep, cm = evaluate(rf, X_test, y_test)
        results['Random Forest'] = acc
        st.markdown("### Random Forest")
        st.write(f"Accuracy: {acc:.2f}")
        st.dataframe(pd.DataFrame(rep).transpose())
        st.pyplot(sns.heatmap(cm, annot=True, fmt='d').figure)

    if "XGBoost" in models_to_run:
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        xgb.fit(X_train, y_train)
        acc, rep, cm = evaluate(xgb, X_test, y_test)
        results['XGBoost'] = acc
        st.markdown("### XGBoost")
        st.write(f"Accuracy: {acc:.2f}")
        st.dataframe(pd.DataFrame(rep).transpose())
        st.pyplot(sns.heatmap(cm, annot=True, fmt='d').figure)

    if "DNN" in models_to_run:
        model = Sequential([
            Dense(128, activation='relu', input_dim=X_train.shape[1]),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(len(np.unique(y_train)), activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(X_train.values, y_train.values, epochs=15, batch_size=32, verbose=0)
        acc = model.evaluate(X_test.values, y_test.values, verbose=0)[1]
        y_pred_dnn = np.argmax(model.predict(X_test.values), axis=1)
        cm = confusion_matrix(y_test, y_pred_dnn)
        report = classification_report(y_test, y_pred_dnn, output_dict=True)
        results['DNN'] = acc
        st.markdown("### Deep Neural Network")
        st.write(f"Accuracy: {acc:.2f}")
        st.dataframe(pd.DataFrame(report).transpose())
        st.pyplot(sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges').figure)

    # Summary
    st.subheader("Summary of Model Accuracies")
    summary_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
    st.table(summary_df)

    csv_report = summary_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download Summary Report", data=csv_report, file_name="model_summary.csv")

st.markdown("---")
st.markdown("""
#### About the Authors
- **Chandrashekar K**, Research Scholar & Developer
- **Dr. Vidya Niranjan**, Professor & Guide
- **Anagha S Setlur**, Research Scholar & Co-developer 

**Affiliation:** Department of Biotechnology, RV College of Engineering, Bangalore

**GitHub Repository:** [View Code](https://github.com/CSK005/Colon-Cancer-AI-ML-model)
""")
