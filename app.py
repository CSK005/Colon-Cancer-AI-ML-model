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
st.title("Colon Cancer Classifier Models)")
st.markdown("""
### Early Detection and Prediction using Machine Learning & Deep Learning
This application provides insights into colon cancer prediction using exome sequencing data.
""")

# File Uploaders for CSVs
uploaded_files = st.file_uploader("Upload your CSV files for training", type=["csv"], accept_multiple_files=True)
uploaded_test_files = st.file_uploader("Upload your CSV files for testing", type=["csv"], accept_multiple_files=True)

# Function to load and preprocess data
def preprocess_data(files):
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
    
    if 'Func.refGene' in df.columns:
        X = df.drop(columns=['Func.refGene'])
        y = df['Func.refGene']
        return X, y
    else:
        st.error("Error: 'Func.refGene' column missing, unable to continue.")
        return None, None

df_train_X, df_train_y = preprocess_data(uploaded_files)
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
