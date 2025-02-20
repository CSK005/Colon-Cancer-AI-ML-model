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
st.title("Colon Cancer Decision Support System (DSS)")
st.markdown("""
### Early Detection and Prediction using Machine Learning & Deep Learning
This application provides insights into colon cancer prediction using exome sequencing data.
""")

# File Uploaders for CSVs
uploaded_files = st.file_uploader("Upload your CSV files for training", type=["csv"], accept_multiple_files=True)
uploaded_test_files = st.file_uploader("Upload your CSV files for testing", type=["csv"], accept_multiple_files=True)

# Required Columns
required_columns = [
    "Chr", "Start", "End", "Ref", "Alt", "Func.refGene", "ExonicFunc.refGene", 
    "CADD", "CADD_Phred", "MutationTaster_score", "MutationAssessor_score", "AF", "AF_popmax"
]

# Function to load data
def load_data(files):
    if files:
        try:
            dfs = [pd.read_csv(file) for file in files]
            df = pd.concat(dfs, ignore_index=True)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    return None

df_train = load_data(uploaded_files)
df_test = load_data(uploaded_test_files)

# Preprocessing Function
def preprocess_data(df):
    if df is None:
        return None, None
    
    # Check for missing columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing columns in dataset: {missing_cols}")
        return None, None
    
    # Label Encoding categorical columns
    categorical_cols = ["Func.refGene", "ExonicFunc.refGene"]
    label_encoders = {}
    for col in categorical_cols:
        df[col] = df[col].astype(str)  # Convert to string
        label_encoders[col] = LabelEncoder()
        df[col] = label_encoders[col].fit_transform(df[col])
    
    # Convert numerical columns
    numerical_cols = ["CADD", "CADD_Phred", "MutationTaster_score", "MutationAssessor_score", "AF", "AF_popmax"]
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric
    df.fillna(0, inplace=True)  # Fill NaN values with 0
    
    # Apply MinMax Scaling
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    
    X = df.drop(columns=['Func.refGene'])
    y = df['Func.refGene']
    return X, y

if df_train is not None and df_test is not None:
    X_train, y_train = preprocess_data(df_train)
    X_test, y_test = preprocess_data(df_test)
    
    if X_train is not None and X_test is not None:
        st.subheader("Training Dataset Overview")
        st.write(df_train.head())
        
        st.subheader("Testing Dataset Overview")
        st.write(df_test.head())

        # RandomForest Model
        st.subheader("RandomForest Classifier")
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)
        y_pred_rf = rf_clf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        st.write(f"RandomForest Accuracy: {accuracy_rf:.2f}")
        st.text(classification_report(y_test, y_pred_rf))
        
        # XGBoost Model
        st.subheader("XGBoost Classifier")
        xgb_clf = XGBClassifier(n_estimators=100, random_state=42)
        xgb_clf.fit(X_train, y_train)
        y_pred_xgb = xgb_clf.predict(X_test)
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        st.write(f"XGBoost Accuracy: {accuracy_xgb:.2f}")
        st.text(classification_report(y_test, y_pred_xgb))
        
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
        model = train_dnn(X_train, y_train)
        _, accuracy_dnn = model.evaluate(X_test, y_test, verbose=0)
        st.write(f"DNN Accuracy: {accuracy_dnn:.2f}")
        
        # Conclusion
        st.subheader("Conclusion & Insights")
        st.markdown(f"""
        - **RandomForest Accuracy:** {accuracy_rf:.2f}
        - **XGBoost Accuracy:** {accuracy_xgb:.2f}
        - **DNN Accuracy:** {accuracy_dnn:.2f}
        - Further improvements can be made with hyperparameter tuning and additional feature selection.
        """)
