import time
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Progress Bar
progress = st.progress(0)
for i in range(100):
    time.sleep(0.02)
    progress.progress(i + 1)

# Sidebar Navigation
menu = st.sidebar.radio("Navigation", ['Home', 'About'])

if menu == "Home":
    st.title("Colon Cancer Decision Support System (DSS)")
    st.subheader("Early Detection and Prediction using ML & DL")
    st.markdown("""
    This application analyzes exome sequencing data to predict colon cancer risk using ML models.
    """)
    
    # File Uploaders
    uploaded_train_files = st.file_uploader("Upload Training CSV Files", type=["csv"], accept_multiple_files=True)
    uploaded_test_files = st.file_uploader("Upload Testing CSV Files", type=["csv"], accept_multiple_files=True)
    
    # Function to Load and Preprocess Data
    def preprocess_data(files):
        if not files:
            return None, None
        try:
            dfs = [pd.read_csv(file) for file in files]
            df = pd.concat(dfs, ignore_index=True)
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None, None

        df.replace(".", np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # Encode categorical columns
        encode_cols = ["Func.refGene", "ExonicFunc.refGene", "Polyphen2_HDIV_pred", "Polyphen2_HVAR_pred", "SIFT_pred"]
        label_encoders = {}
        for col in encode_cols:
            if col in df.columns:
                label_encoders[col] = LabelEncoder()
                df[col] = label_encoders[col].fit_transform(df[col].astype(str))

        # Normalize selected columns
        scale_cols = ["CADD", "CADD_Phred", "MutationTaster_score", "AF", "AF_popmax"]
        scaler = MinMaxScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

        if 'Func.refGene' in df.columns:
            X = df.drop(columns=['Func.refGene'])
            y = df['Func.refGene']
            return X, y
        else:
            st.error("Missing 'Func.refGene' column.")
            return None, None

    df_train_X, df_train_y = preprocess_data(uploaded_train_files)
    df_test_X, df_test_y = preprocess_data(uploaded_test_files)

    if df_train_X is not None and df_test_X is not None:
        st.subheader("Training Dataset Overview")
        st.write(df_train_X.head())

        st.subheader("Testing Dataset Overview")
        st.write(df_test_X.head())

        # Model Training and Evaluation
        models = {
            "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
            "XGBoost": XGBClassifier(n_estimators=100, random_state=42)
        }
        
        results = {}
        for name, model in models.items():
            try:
                st.subheader(f"{name} Classifier")
                model.fit(df_train_X, df_train_y)
                y_pred = model.predict(df_test_X)
                accuracy = accuracy_score(df_test_y, y_pred)
                results[name] = accuracy
                st.write(f"{name} Accuracy: {accuracy:.2f}")
                st.text(classification_report(df_test_y, y_pred))
            except Exception as e:
                st.error(f"Error training {name}: {e}")

        # Deep Learning Model
        try:
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
            dnn_model = train_dnn(df_train_X, df_train_y)
            _, accuracy_dnn = dnn_model.evaluate(df_test_X, df_test_y, verbose=0)
            results["DNN"] = accuracy_dnn
            st.write(f"DNN Accuracy: {accuracy_dnn:.2f}")
        except Exception as e:
            st.error(f"Error training DNN model: {e}")
        
        # Summary
        st.subheader("Conclusion & Insights")
        for model, acc in results.items():
            st.markdown(f"- **{model} Accuracy:** {acc:.2f}")

if menu == "About":
    st.title("About This Project")
    st.markdown("""
    This application provides a decision support system for colon cancer prediction using exome sequencing data.
    """)
