import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Title and Introduction
st.title("Colon Cancer Classifier Models")
st.markdown("""
### Early Detection and Prediction using Machine Learning & Deep Learning  
This application provides insights into colon cancer prediction using exome sequencing data.
""")

# Example Dataset Display
st.subheader("Example Dataset Format")
example_data = {
    'Chr': ['chr1', 'chr2', 'chrX', 'chr3', 'chrY'],
    'Start': [12345, 67890, 13579, 24680, 11223],
    'End': [12355, 67900, 13589, 24690, 11233],
    'Func.refGene': ['exonic', 'intronic', 'splicing', 'exonic', 'intronic'],
    'ExonicFunc.refGene': ['nonsynonymous', 'synonymous', 'nonsynonymous', 'synonymous', 'nonsynonymous'],
    'Polyphen2_HDIV_pred': ['benign', 'possibly_damaging', 'probably_damaging', 'benign', 'benign'],
    'Polyphen2_HVAR_pred': ['benign', 'benign', 'probably_damaging', 'benign', 'possibly_damaging'],
    'SIFT_pred': ['tolerated', 'deleterious', 'deleterious', 'tolerated', 'tolerated'],
    'MutationTaster_pred': ['disease_causing', 'polymorphism', 'disease_causing', 'polymorphism', 'polymorphism'],
    'MutationAssessor_pred': ['low', 'medium', 'high', 'low', 'medium'],
    'CLNSIG': ['pathogenic', 'benign', 'pathogenic', 'benign', 'uncertain'],
    'CADD': [12.3, 23.5, 35.7, 10.1, 5.6],
    'CADD_Phred': [20.1, 25.3, 30.7, 15.4, 10.2],
    'MutationTaster_score': [0.8, 0.2, 0.9, 0.1, 0.3],
    'MutationAssessor_score': [1.2, 2.3, 3.4, 1.1, 0.9],
    'AF': [0.01, 0.05, 0.001, 0.02, 0.03],
    'AF_popmax': [0.02, 0.06, 0.002, 0.03, 0.04]
}
example_df = pd.DataFrame(example_data)
st.dataframe(example_df)

# File Uploaders for CSVs
uploaded_files = st.file_uploader("Upload your CSV files for training", type=["csv"], accept_multiple_files=True)
uploaded_test_files = st.file_uploader("Upload your CSV files for testing", type=["csv"], accept_multiple_files=True)

def preprocess_data(files):
    if not files:
        return None, None
    
    try:
        dfs = [pd.read_csv(file) for file in files]
        df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return None, None
    
    # Replace '.' with NaN and fill missing values
    df.replace(".", np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Convert chromosome names to numeric
    chr_map = {f'chr{i}': i for i in range(1, 23)}
    chr_map.update({'chrX': 23, 'chrY': 24})
    df['Chr'] = df['Chr'].map(chr_map).fillna(0).astype(int)

    # Columns to encode
    encode_cols = ["Func.refGene", "ExonicFunc.refGene", "Polyphen2_HDIV_pred", "Polyphen2_HVAR_pred",
                   "SIFT_pred", "MutationTaster_pred", "MutationAssessor_pred", "CLNSIG"]
    
    label_encoders = {}
    for col in encode_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
        else:
            st.warning(f"Column '{col}' not found and will be skipped.")

    # Convert remaining object columns to numeric
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    # Normalize numerical columns
    scale_cols = ["CADD", "CADD_Phred", "MutationTaster_score", "MutationAssessor_score", "AF", "AF_popmax"]
    for col in scale_cols:
        if col not in df.columns:
            st.warning(f"Column '{col}' not found for scaling.")
    scale_cols = [col for col in scale_cols if col in df.columns]
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])

    if 'Func.refGene' not in df.columns:
        st.error("Target column 'Func.refGene' is missing.")
        return None, None

    X = df.drop(columns=['Func.refGene'])
    y = df['Func.refGene']
    return X, y

df_train_X, df_train_y = preprocess_data(uploaded_files)
df_test_X, df_test_y = preprocess_data(uploaded_test_files)

if df_train_X is not None and df_test_X is not None:
    st.subheader("Training Dataset Preview")
    st.dataframe(df_train_X.head())

    st.subheader("Testing Dataset Preview")
    st.dataframe(df_test_X.head())

    # Visualize feature distributions
    st.subheader("Feature Distributions")
    numeric_features = df_train_X.select_dtypes(include=[np.number]).columns.tolist()
    selected_feature = st.selectbox("Select a numeric feature to visualize", numeric_features)
    fig, ax = plt.subplots()
    sns.histplot(df_train_X[selected_feature], bins=30, kde=True, ax=ax)
    ax.set_title(f"Distribution of {selected_feature} in Training Data")
    st.pyplot(fig)

    # Train and evaluate models
    try:
        st.subheader("Random Forest Classifier")
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(df_train_X, df_train_y)
        y_pred_rf = rf_clf.predict(df_test_X)
        acc_rf = accuracy_score(df_test_y, y_pred_rf)
        st.write(f"Accuracy: {acc_rf:.2f}")

        # Confusion matrix
        cm_rf = confusion_matrix(df_test_y, y_pred_rf)
        fig, ax = plt.subplots()
        sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Random Forest Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        # Classification report visualization (bar plot of precision, recall, f1-score)
        report_rf = classification_report(df_test_y, y_pred_rf, output_dict=True)
        metrics_df = pd.DataFrame(report_rf).transpose().iloc[:-3, :3]  # exclude avg rows
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics_df.plot(kind='bar', ax=ax)
        ax.set_title("Random Forest Classification Metrics by Class")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Random Forest training or evaluation failed: {e}")

    try:
        st.subheader("XGBoost Classifier")
        xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, random_state=42)
        xgb_clf.fit(df_train_X, df_train_y)
        y_pred_xgb = xgb_clf.predict(df_test_X)
        acc_xgb = accuracy_score(df_test_y, y_pred_xgb)
        st.write(f"Accuracy: {acc_xgb:.2f}")

        cm_xgb = confusion_matrix(df_test_y, y_pred_xgb)
        fig, ax = plt.subplots()
        sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', ax=ax)
        ax.set_title("XGBoost Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        report_xgb = classification_report(df_test_y, y_pred_xgb, output_dict=True)
        metrics_df = pd.DataFrame(report_xgb).transpose().iloc[:-3, :3]
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics_df.plot(kind='bar', ax=ax)
        ax.set_title("XGBoost Classification Metrics by Class")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"XGBoost training or evaluation failed: {e}")

    try:
        st.subheader("Deep Neural Network (DNN)")

        @st.cache_resource
        def train_dnn(X_train, y_train):
            model = Sequential([
                Dense(128, input_dim=X_train.shape[1], activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(len(np.unique(y_train)), activation='softmax')
            ])
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=0)
            return model

        dnn_model = train_dnn(df_train_X.values, df_train_y.values)
        loss, acc_dnn = dnn_model.evaluate(df_test_X.values, df_test_y.values, verbose=0)
        st.write(f"Accuracy: {acc_dnn:.2f}")

        # Predict and prepare classification report
        y_pred_dnn_prob = dnn_model.predict(df_test_X.values)
        y_pred_dnn = np.argmax(y_pred_dnn_prob, axis=1)
        cm_dnn = confusion_matrix(df_test_y, y_pred_dnn)
        fig, ax = plt.subplots()
        sns.heatmap(cm_dnn, annot=True, fmt='d', cmap='Oranges', ax=ax)
        ax.set_title("DNN Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig)

        report_dnn = classification_report(df_test_y, y_pred_dnn, output_dict=True)
        metrics_df = pd.DataFrame(report_dnn).transpose().iloc[:-3, :3]
        fig, ax = plt.subplots(figsize=(8, 4))
        metrics_df.plot(kind='bar', ax=ax)
        ax.set_title("DNN Classification Metrics by Class")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

    except Exception as e:
        st.error(f"DNN training or evaluation failed: {e}")

    # Summary
    st.subheader("Summary of Model Accuracies")
    summary_df = pd.DataFrame({
        'Model': ['Random Forest', 'XGBoost', 'DNN'],
        'Accuracy': [acc_rf if 'acc_rf' in locals() else None,
                     acc_xgb if 'acc_xgb' in locals() else None,
                     acc_dnn if 'acc_dnn' in locals() else None]
    })
    st.table(summary_df)
else:
    st.info("Please upload both training and testing CSV files to proceed.")
