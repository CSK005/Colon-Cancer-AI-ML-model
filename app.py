import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

def load_data(uploaded_files):
    if uploaded_files:
        try:
            dfs = [pd.read_csv(file) for file in uploaded_files]
            df = pd.concat(dfs, ignore_index=True)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    else:
        st.warning("Please upload CSV files to proceed.")
        return None

df_train = load_data(uploaded_files)
df_test = load_data(uploaded_test_files)

if df_train is not None and df_test is not None:
    st.subheader("Training Dataset Overview")
    st.write(df_train.head())
    
    st.subheader("Testing Dataset Overview")
    st.write(df_test.head())

    # Ensure required column exists
    if 'Gene.refGeneWithVer' not in df_train.columns or 'Gene.refGeneWithVer' not in df_test.columns:
        st.error("Error: The required column 'Gene.refGeneWithVer' is missing from one of the datasets.")
    else:
        # Preprocessing
        st.subheader("Data Preprocessing")
        st.write("Feature selection and encoding applied.")

        def preprocess_data(df):
            df = df.select_dtypes(include=[np.number]).fillna(0)
            X = df.drop(columns=['Gene.refGeneWithVer'])
            y = df['Gene.refGeneWithVer']
            return X, y

        X_train, y_train = preprocess_data(df_train)
        X_test, y_test = preprocess_data(df_test)

        # RandomForest Model
        st.subheader("RandomForest Classifier")
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)
        y_pred_rf = rf_clf.predict(X_test)
        accuracy_rf = accuracy_score(y_test, y_pred_rf)
        st.write(f"RandomForest Accuracy: {accuracy_rf:.2f}")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred_rf))

        # Confusion Matrix - RF
        def plot_confusion_matrix(conf_matrix, title):
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            plt.title(title)
            st.pyplot(fig)

        st.subheader("Confusion Matrix - RandomForest")
        conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
        plot_confusion_matrix(conf_matrix_rf, "RandomForest Confusion Matrix")

        # Feature Importance - RF
        st.subheader("Feature Importance - RandomForest")
        feature_importances = rf_clf.feature_importances_
        features = df_train.drop(columns=['Gene.refGeneWithVer']).columns
        importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
        importance_df = importance_df.sort_values(by='Importance', ascending=False)
        fig, ax = plt.subplots()
        sns.barplot(x='Importance', y='Feature', data=importance_df, ax=ax)
        plt.title("Feature Importance - RandomForest")
        st.pyplot(fig)

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

        y_pred_dnn = np.argmax(model.predict(X_test), axis=-1)
        st.text("DNN Classification Report:")
        st.text(classification_report(y_test, y_pred_dnn))

        # Confusion Matrix - DNN
        st.subheader("Confusion Matrix - DNN")
        conf_matrix_dnn = confusion_matrix(y_test, y_pred_dnn)
        plot_confusion_matrix(conf_matrix_dnn, "DNN Confusion Matrix")

        # Conclusion
        st.subheader("Conclusion & Insights")
        st.markdown(f"""
        - **RandomForest performed with an accuracy of** {accuracy_rf:.2f}
        - **DNN achieved an accuracy of** {accuracy_dnn:.2f}
        - Further improvements can be made with hyperparameter tuning and additional feature selection.
        """)
