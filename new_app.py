import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Title and Introduction
st.title("Colon Cancer Decision Support System (DSS)")
st.markdown("""
### Early Detection and Prediction using Machine Learning & Deep Learning
This application provides insights into colon cancer prediction using exome sequencing data.
""")

# File Uploaders for CSVs
normal_files = st.file_uploader("Upload your CSV files for normal exome datasets", type=["csv"], accept_multiple_files=True)
cancer_files = st.file_uploader("Upload your CSV files for colon cancer exome datasets", type=["csv"], accept_multiple_files=True)
uploaded_test_files = st.file_uploader("Upload your CSV files for testing", type=["csv"], accept_multiple_files=True)

# Function to load and preprocess data
def preprocess_data(files, label=None):
    if not files:
        return None, None
    
    try:
        dfs = [pd.read_csv(file) for file in files]
        df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None
    
    # Replace '.' with NaN and handle missing values
    df.replace(".", np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Convert chromosome numbers
    df['Chr'] = df['Chr'].replace({'chr1': 1, 'chr2': 2, 'chr3': 3, 'chr4': 4, 'chr5': 5,
                                   'chr6': 6, 'chr7': 7, 'chr8': 8, 'chr9': 9, 'chr10': 10,
                                   'chr11': 11, 'chr12': 12, 'chr13': 13, 'chr14': 14, 'chr15': 15,
                                   'chr16': 16, 'chr17': 17, 'chr18': 18, 'chr19': 19, 'chr20': 20,
                                   'chr21': 21, 'chr22': 22, 'chrX': 23, 'chrY': 24,
                                   'chr7_gl000195_random': 25})  # Handle other random chromosomes similarly
    
    # Columns to encode
    encode_cols = ["Func.refGene", "ExonicFunc.refGene", "Polyphen2_HDIV_pred", "Polyphen2_HVAR_pred",
                   "SIFT_pred", "MutationTaster_pred", "MutationAssessor_pred", "CLNSIG"]
    
    # Label Encoding categorical columns
    label_encoders = {}
    for col in encode_cols:
        if col in df.columns:
            label_encoders[col] = LabelEncoder()
            df[col] = label_encoders[col].fit_transform(df[col].astype(str))
        else:
            st.warning(f"Column {col} not found in dataset and will be skipped.")
    
    # Convert object columns to numeric
    object_cols = df.select_dtypes(include=['object']).columns
    for col in object_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Normalize numerical columns
    scale_cols = ["CADD", "CADD_Phred", "MutationTaster_score", "MutationAssessor_score", "AF", "AF_popmax"]
    
    scaler = MinMaxScaler()
    df[scale_cols] = scaler.fit_transform(df[scale_cols])
    
    if label is not None:
        df['Label'] = label
    
    X = df.drop(columns=['Label']) if 'Label' in df.columns else df
    y = df['Label'] if 'Label' in df.columns else None
    return X, y

# Preprocess normal and cancer datasets
df_normal_X, df_normal_y = preprocess_data(normal_files, label=0)
df_cancer_X, df_cancer_y = preprocess_data(cancer_files, label=1)

# Combine normal and cancer datasets for training
if df_normal_X is not None and df_cancer_X is not None:
    df_train_X = pd.concat([df_normal_X, df_cancer_X], ignore_index=True)
    df_train_y = pd.concat([df_normal_y, df_cancer_y], ignore_index=True)

# Preprocess test datasets
df_test_X, df_test_y = preprocess_data(uploaded_test_files)

if df_train_X is not None and df_test_X is not None:
    st.subheader("Training Dataset Overview")
    st.write(df_train_X.head())
    
    st.subheader("Testing Dataset Overview")
    st.write(df_test_X.head())

    try:
        # RandomForest Model
        st.subheader("RandomForest Classifier")
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(df_train_X, df_train_y)
        y_pred_rf = rf_clf.predict(df_test_X)
        accuracy_rf = accuracy_score(df_test_y, y_pred_rf)
        st.write(f"RandomForest Accuracy: {accuracy_rf:.2f}")
        st.text(classification_report(df_test_y, y_pred_rf))
        
        # ROC Curve for RandomForest
        y_prob_rf = rf_clf.predict_proba(df_test_X)[:, 1]
        fpr_rf, tpr_rf, _ = roc_curve(df_test_y, y_prob_rf)
        roc_auc_rf = roc_auc_score(df_test_y, y_prob_rf)
        plt.figure()
        plt.plot(fpr_rf, tpr_rf, label=f'RandomForest (AUC = {roc_auc_rf:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error training RandomForest model: {e}")
    
    try:
        # XGBoost Model
        st.subheader("XGBoost Classifier")
        xgb_clf = XGBClassifier(n_estimators=100, random_state=42)
        xgb_clf.fit(df_train_X, df_train_y)
        y_pred_xgb = xgb_clf.predict(df_test_X)
        accuracy_xgb = accuracy_score(df_test_y, y_pred_xgb)
        st.write(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
        st.text(classification_report(df_test_y, y_pred_xgb))
        
        # ROC Curve for XGBoost
        y_prob_xgb = xgb_clf.predict_proba(df_test_X)[:, 1]
        fpr_xgb, tpr_xgb, _ = roc_curve(df_test_y, y_prob_xgb)
        roc_auc_xgb = roc_auc_score(df_test_y, y_prob_xgb)
        plt.figure()
        plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error training XGBoost model: {e}")
    
    try:
        # DNN Model (Cached for Performance)
        @st.cache_resource
        def train_dnn(X_train, y_train):
            model = Sequential([
                Dense(128, input_dim=X_train.shape[1], activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(len(np.unique(y_train)), activation='softmax')
            ])
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            return model
    
        st.subheader("Deep Neural Network (DNN)")
        model = train_dnn(df_train_X, df_train_y)
        _, accuracy_dnn = model.evaluate(df_test_X, df_test_y, verbose=0)
        st.write(f"DNN Accuracy: {accuracy_dnn:.2f}")

        # ROC Curve for DNN
        y_prob_dnn = model.predict(df_test_X)[:, 1]
        fpr_dnn, tpr_dnn, _ = roc_curve(df_test_y, y_prob_dnn)
        roc_auc_dnn = roc_auc_score(df_test_y, y_prob_dnn)
        plt.figure()
        plt.plot(fpr_dnn, tpr_dnn, label=f'DNN (AUC = {roc_auc_dnn:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error training DNN model: {e}")
    
    # Conclusion
    st.subheader("Conclusion & Insights")
    st.markdown(f"""
    - **RandomForest Accuracy:** {accuracy_rf:.2f}
    - **XGBoost Accuracy:** {accuracy_xgb:.2f}
    - **DNN Accuracy:** {accuracy_dnn:.2f}
    - Further improvements can be made with hyperparameter tuning and additional feature selection.
    """)

    # Additional Models for Cancer Progression, Recurrence, Risk, and Relapse Prediction
    
    st.subheader("Additional Models")
    
    try:
        # Cancer Progression Prediction Model (GradientBoostingClassifier)
        st.subheader("Cancer Progression Prediction - GradientBoostingClassifier")
        gbc = GradientBoostingClassifier(n_estimators=100, random_state=42)
        gbc.fit(df_train_X, df_train_y)
        y_pred_gbc = gbc.predict(df_test_X)
        accuracy_gbc = accuracy_score(df_test_y, y_pred_gbc)
        st.write(f"GradientBoostingClassifier Accuracy: {accuracy_gbc:.2f}")
        st.text(classification_report(df_test_y, y_pred_gbc))
        
        # ROC Curve for GradientBoostingClassifier
        y_prob_gbc = gbc.predict_proba(df_test_X)[:, 1]
        fpr_gbc, tpr_gbc, _ = roc_curve(df_test_y, y_prob_gbc)
        roc_auc_gbc = roc_auc_score(df_test_y, y_prob_gbc)
        plt.figure()
        plt.plot(fpr_gbc, tpr_gbc, label=f'GradientBoosting (AUC = {roc_auc_gbc:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error training GradientBoostingClassifier model: {e}")
    
    try:
        # Cancer Recurrence Prediction Model (RandomForestClassifier)
        st.subheader("Cancer Recurrence Prediction - RandomForestClassifier")
        rf_recurrence = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_recurrence.fit(df_train_X, df_train_y)
        y_pred_rf_recurrence = rf_recurrence.predict(df_test_X)
        accuracy_rf_recurrence = accuracy_score(df_test_y, y_pred_rf_recurrence)
        st.write(f"RandomForestClassifier Accuracy: {accuracy_rf_recurrence:.2f}")
        st.text(classification_report(df_test_y, y_pred_rf_recurrence))
        
        # ROC Curve for RandomForestClassifier (Recurrence)
        y_prob_rf_recurrence = rf_recurrence.predict_proba(df_test_X)[:, 1]
        fpr_rf_recurrence, tpr_rf_recurrence, _ = roc_curve(df_test_y, y_prob_rf_recurrence)
        roc_auc_rf_recurrence = roc_auc_score(df_test_y, y_prob_rf_recurrence)
        plt.figure()
        plt.plot(fpr_rf_recurrence, tpr_rf_recurrence, label=f'RandomForest Recurrence (AUC = {roc_auc_rf_recurrence:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error training Recurrence RandomForest model: {e}")
    
    try:
        # Cancer Risk Prediction Model (XGBoostClassifier)
        st.subheader("Cancer Risk Prediction - XGBoostClassifier")
        xgb_risk = XGBClassifier(n_estimators=100, random_state=42)
        xgb_risk.fit(df_train_X, df_train_y)
        y_pred_xgb_risk = xgb_risk.predict(df_test_X)
        accuracy_xgb_risk = accuracy_score(df_test_y, y_pred_xgb_risk)
        st.write(f"XGBoostClassifier Accuracy: {accuracy_xgb_risk:.2f}")
        st.text(classification_report(df_test_y, y_pred_xgb_risk))
        
        # ROC Curve for XGBoostClassifier (Risk)
        y_prob_xgb_risk = xgb_risk.predict_proba(df_test_X)[:, 1]
        fpr_xgb_risk, tpr_xgb_risk, _ = roc_curve(df_test_y, y_prob_xgb_risk)
        roc_auc_xgb_risk = roc_auc_score(df_test_y, y_prob_xgb_risk)
        plt.figure()
        plt.plot(fpr_xgb_risk, tpr_xgb_risk, label=f'XGBoost Risk (AUC = {roc_auc_xgb_risk:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error training Risk XGBoost model: {e}")
    
    try:
        # Cancer Relapse Prediction Model (DNN)
        st.subheader("Cancer Relapse Prediction - DNN")
        model_relapse = train_dnn(df_train_X, df_train_y)
        _, accuracy_dnn_relapse = model_relapse.evaluate(df_test_X, df_test_y, verbose=0)
        st.write(f"DNN Relapse Prediction Accuracy: {accuracy_dnn_relapse:.2f}")

        # ROC Curve for DNN (Relapse)
        y_prob_dnn_relapse = model_relapse.predict(df_test_X)[:, 1]
        fpr_dnn_relapse, tpr_dnn_relapse, _ = roc_curve(df_test_y, y_prob_dnn_relapse)
        roc_auc_dnn_relapse = roc_auc_score(df_test_y, y_prob_dnn_relapse)
        plt.figure()
        plt.plot(fpr_dnn_relapse, tpr_dnn_relapse, label=f'DNN Relapse (AUC = {roc_auc_dnn_relapse:.2f})')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='lower right')
        st.pyplot(plt)
    except Exception as e:
        st.error(f"Error training Relapse DNN model: {e}")
    
    # Feature Correlation Graph
    st.subheader("Feature Correlation Graph")
    corr = df_train_X.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, cmap='coolwarm', annot=True)
    plt.title("Feature Correlation Matrix")
    st.pyplot(plt)
