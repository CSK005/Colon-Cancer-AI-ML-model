import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

# Streamlit App Title
st.title("Colon Cancer DSS: Data Visualization & ML Insights")

# File Uploaders for Training & Testing Datasets
uploaded_train_files = st.file_uploader("Upload CSV files for training", type=["csv"], accept_multiple_files=True)
uploaded_test_files = st.file_uploader("Upload CSV files for testing", type=["csv"], accept_multiple_files=True)

# Function to Load & Preprocess Data
def preprocess_data(files):
    if not files:
        return None, None

    try:
        dfs = [pd.read_csv(file) for file in files]
        df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None, None

    # Replace "." with NaN
    df.replace(".", np.nan, inplace=True)

    # Fill missing values only for numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Encode Chromosome Numbers
    if "Chr" in df.columns:
        df["Chr"] = df["Chr"].map({
            'chr1': 1, 'chr2': 2, 'chr3': 3, 'chr4': 4, 'chr5': 5,
            'chr6': 6, 'chr7': 7, 'chr8': 8, 'chr9': 9, 'chr10': 10,
            'chr11': 11, 'chr12': 12, 'chr13': 13, 'chr14': 14, 'chr15': 15,
            'chr16': 16, 'chr17': 17, 'chr18': 18, 'chr19': 19, 'chr20': 20,
            'chr21': 21, 'chr22': 22, 'chrX': 23, 'chrY': 24
        }).fillna(0)

    # Encode Categorical Columns
    ordered_mappings = {
        "Polyphen2_HDIV_pred": {"D": 2, "P": 1, "B": 0},
        "Polyphen2_HVAR_pred": {"D": 2, "P": 1, "B": 0},
        "SIFT_pred": {"D": 2, "T": 1},
        "MutationTaster_pred": {"A": 3, "D": 2, "N": 1, "P": 0},
        "CLNSIG": {"Pathogenic": 2, "Likely_pathogenic": 1, "Benign": 0}
    }
    
    for col, mapping in ordered_mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0)

    # Normalize Selected Features
    scale_cols = ["CADD", "CADD_Phred", "MutationTaster_score", "MutationAssessor_score", "AF", "AF_popmax"]
    if any(col in df.columns for col in scale_cols):
        scaler = MinMaxScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    # Check if Target Column Exists
    if "Func.refGene" in df.columns:
        X = df.drop(columns=["Func.refGene"])
        y = df["Func.refGene"]
        return X, y
    else:
        st.error("Error: 'Func.refGene' column missing in dataset.")
        return None, None

# Load & Preprocess Training & Testing Data
df_train_X, df_train_y = preprocess_data(uploaded_train_files)
df_test_X, df_test_y = preprocess_data(uploaded_test_files)

if df_train_X is not None and df_test_X is not None:
    st.subheader("Training Dataset Overview")
    st.write(df_train_X.head())

    # Encode Target Variable
    label_encoder = LabelEncoder()
    df_train_y = label_encoder.fit_transform(df_train_y)
    df_test_y = label_encoder.transform(df_test_y)

    # Display Class Distribution Before SMOTE
    st.subheader("Class Distribution in Training Data")
    class_counts = pd.Series(df_train_y).value_counts()
    st.write(class_counts)

    # Apply SMOTE Only If More Than One Class Exists
    if len(class_counts) > 1:
        smote = SMOTE(random_state=42)
        df_train_X, df_train_y = smote.fit_resample(df_train_X, df_train_y)
    else:
        st.warning("SMOTE not applied: Only one class found in the training dataset.")

    # Train Models
    rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    rf_clf.fit(df_train_X, df_train_y)
    y_pred_rf = rf_clf.predict(df_test_X)
    accuracy_rf = accuracy_score(df_test_y, y_pred_rf)

    xgb_clf = XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    xgb_clf.fit(df_train_X, df_train_y)
    y_pred_xgb = xgb_clf.predict(df_test_X)
    accuracy_xgb = accuracy_score(df_test_y, y_pred_xgb)

    # Display Model Accuracy
    st.subheader("Model Performance")
    st.write(f"**RandomForest Accuracy:** {accuracy_rf:.2f}")
    st.write(f"**XGBoost Accuracy:** {accuracy_xgb:.2f}")

    # Confusion Matrix - RandomForest
    st.subheader("Confusion Matrix - RandomForest")
    cm_rf = confusion_matrix(df_test_y, y_pred_rf)
    fig, ax = plt.subplots()
    sns.heatmap(cm_rf, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Confusion Matrix - XGBoost
    st.subheader("Confusion Matrix - XGBoost")
    cm_xgb = confusion_matrix(df_test_y, y_pred_xgb)
    fig, ax = plt.subplots()
    sns.heatmap(cm_xgb, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Feature Importance - RandomForest
    st.subheader("Feature Importance - RandomForest")
    rf_feature_importances = pd.DataFrame({'Feature': df_train_X.columns, 'Importance': rf_clf.feature_importances_})
    rf_feature_importances = rf_feature_importances.sort_values(by='Importance', ascending=False)
    fig, ax = plt.subplots()
    sns.barplot(data=rf_feature_importances[:10], x='Importance', y='Feature', ax=ax)
    st.pyplot(fig)

    # Mutation Distribution Plot
    st.subheader("Mutation Distribution Across Samples")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.histplot(df_train_y, kde=True, bins=20, ax=ax)
    plt.xlabel("Mutation Categories")
    plt.ylabel("Frequency")
    st.pyplot(fig)

    # Conclusion
    st.subheader("Conclusion & Insights")
    st.markdown(f"""
    - **RandomForest Accuracy:** {accuracy_rf:.2f}
    - **XGBoost Accuracy:** {accuracy_xgb:.2f}
    - Feature importance analysis suggests **{rf_feature_importances.iloc[0, 0]}** as a key predictor.
    """)

