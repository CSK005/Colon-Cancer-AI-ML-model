import streamlit as st
import pandas as pd
import numpy as np

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

# File Uploader for CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error loading file: {e}")
            return None
    else:
        st.warning("Please upload a CSV file to proceed.")
        return None

df = load_data(uploaded_file)
if df is not None:
    st.subheader("Dataset Overview")
    st.write(df.head())

    # Ensure required column exists
    if 'Gene.refGeneWithVer' not in df.columns:
        st.error("Error: The required column 'Gene.refGeneWithVer' is missing from the dataset.")
    else:
        # Preprocessing
        st.subheader("Data Preprocessing")
        st.write("Feature selection and encoding applied.")

        def preprocess_data(df):
            df = df.select_dtypes(include=[np.number]).fillna(0)
            X = df.drop(columns=['Gene.refGeneWithVer'])
            y = df['Gene.refGeneWithVer']
            return train_test_split(X, y, test_size=0.2, random_state=42)

        X_train, X_test, y_train, y_test = preprocess_data(df)

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
        def plot_confusion_matrix(conf_matrix):
            import matplotlib.pyplot as plt  # Lazy import
            import seaborn as sns  # Lazy import
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)

        st.subheader("Confusion Matrix - RandomForest")
        conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
        plot_confusion_matrix(conf_matrix_rf)

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
        plot_confusion_matrix(conf_matrix_dnn)

        # Conclusion
        st.subheader("Conclusion & Insights")
        st.markdown(f"""
        - **RandomForest performed with an accuracy of** {accuracy_rf:.2f}
        - **DNN achieved an accuracy of** {accuracy_dnn:.2f}
        - Further improvements can be made with hyperparameter tuning and additional feature selection.
        """)
