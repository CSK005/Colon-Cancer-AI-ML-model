# app.py - Shiny for Python: Colon Cancer Classifier

from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# --------------------
# UI
# --------------------
app_ui = ui.page_fluid(
    ui.h2("Colon Cancer Classifier Models"),
    ui.p("Early Detection and Prediction using Machine Learning & Deep Learning"),

    ui.layout_sidebar(
        ui.panel_sidebar(
            ui.input_file("train_file", "Upload Training CSVs", multiple=True, accept=".csv"),
            ui.input_file("test_file", "Upload Testing CSVs", multiple=True, accept=".csv"),
            ui.input_select("feature_select", "Select feature to visualize", choices=[]),
            ui.input_action_button("run_models", "Run Models"),
        ),
        ui.panel_main(
            ui.output_table("train_preview"),
            ui.output_table("test_preview"),
            ui.output_plot("feature_plot"),
            ui.output_text("rf_acc"),
            ui.output_plot("rf_cm"),
            ui.output_text("xgb_acc"),
            ui.output_plot("xgb_cm"),
            ui.output_text("dnn_acc"),
            ui.output_plot("dnn_cm"),
        )
    ),

    ui.hr(),
    ui.h3("About Us"),
    ui.p("This Colon Cancer classification model was developed to provide insights "
         "into colon cancer prediction using exome sequencing data."),
    ui.a("GitHub Repository", href="https://github.com/CSK005/Colon-Cancer-AI-ML-model"),
    ui.tags.ul(
        ui.tags.li("CHANDRASHEKAR K"),
        ui.tags.li("Dr. VIDYA NIRANJAN"),
        ui.tags.li("ANAGHA S SETLUR"),
    )
)

# --------------------
# Data Preprocessing
# --------------------
def preprocess(files):
    if not files:
        return None, None

    dfs = [pd.read_csv(f["datapath"]) for f in files]
    df = pd.concat(dfs, ignore_index=True)

    # Cleaning
    df.replace(".", np.nan, inplace=True)
    df.fillna(0, inplace=True)

    # Encode Chr column if present
    chr_map = {f'chr{i}': i for i in range(1, 23)}
    chr_map.update({'chrX': 23, 'chrY': 24})
    if 'Chr' in df.columns:
        df['Chr'] = df['Chr'].map(chr_map).fillna(0).astype(int)

    # Encode categorical columns
    encode_cols = [
        "Func.refGene", "ExonicFunc.refGene", "Polyphen2_HDIV_pred",
        "Polyphen2_HVAR_pred", "SIFT_pred", "MutationTaster_pred",
        "MutationAssessor_pred", "CLNSIG"
    ]
    for col in encode_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

    # Normalize numeric columns
    scale_cols = ["CADD", "CADD_Phred", "MutationTaster_score",
                  "MutationAssessor_score", "AF", "AF_popmax"]
    scale_cols = [c for c in scale_cols if c in df.columns]
    if scale_cols:
        scaler = MinMaxScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])

    if 'Func.refGene' not in df.columns:
        return None, None

    X = df.drop(columns=['Func.refGene'])
    y = df['Func.refGene']
    return X, y

# --------------------
# SERVER
# --------------------
def server(input, output, session):

    train_data = reactive.Value(None)
    test_data = reactive.Value(None)

    # Load Data
    @reactive.Effect
    @reactive.event(input.train_file, input.test_file)
    def _load_data():
        X_train, y_train = preprocess(input.train_file())
        X_test, y_test = preprocess(input.test_file())
        if X_train is not None and X_test is not None:
            train_data.set((X_train, y_train))
            test_data.set((X_test, y_test))

    # Preview Tables
    @output
    @render.table
    def train_preview():
        if train_data():
            return train_data()[0].head()

    @output
    @render.table
    def test_preview():
        if test_data():
            return test_data()[0].head()

    # Feature Distribution Plot
    @output
    @render.plot
    def feature_plot():
        if train_data():
            df = train_data()[0]
            feature = input.feature_select()
            if feature and feature in df.columns:
                fig, ax = plt.subplots()
                sns.histplot(df[feature], bins=30, kde=True, ax=ax)
                ax.set_title(f"Distribution of {feature}")
                return fig

    # Run Models
    @reactive.Effect
    @reactive.event(input.run_models)
    def _train_models():
        if not (train_data() and test_data()):
            return

        X_train, y_train = train_data()
        X_test, y_test = test_data()

        # ---- Random Forest ----
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred_rf = rf.predict(X_test)
        acc_rf = accuracy_score(y_test, y_pred_rf)

        @output
        @render.text
        def rf_acc():
            return f"Random Forest Accuracy: {acc_rf:.2f}"

        @output
        @render.plot
        def rf_cm():
            cm = confusion_matrix(y_test, y_pred_rf)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
            ax.set_title("Random Forest Confusion Matrix")
            return fig

        # ---- XGBoost ----
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
        xgb.fit(X_train, y_train)
        y_pred_xgb = xgb.predict(X_test)
        acc_xgb = accuracy_score(y_test, y_pred_xgb)

        @output
        @render.text
        def xgb_acc():
            return f"XGBoost Accuracy: {acc_xgb:.2f}"

        @output
        @render.plot
        def xgb_cm():
            cm = confusion_matrix(y_test, y_pred_xgb)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Greens", ax=ax)
            ax.set_title("XGBoost Confusion Matrix")
            return fig

        # ---- Deep Neural Network ----
        model = Sequential([
            Dense(64, input_dim=X_train.shape[1], activation='relu'),
            Dense(32, activation='relu'),
            Dense(len(np.unique(y_train)), activation='softmax')
        ])
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
        model.fit(X_train, y_train, epochs=3, batch_size=32, verbose=0)

        _, acc_dnn = model.evaluate(X_test, y_test, verbose=0)

        @output
        @render.text
        def dnn_acc():
            return f"DNN Accuracy: {acc_dnn:.2f}"

        @output
        @render.plot
        def dnn_cm():
            y_pred_prob = model.predict(X_test, verbose=0)
            y_pred_dnn = np.argmax(y_pred_prob, axis=1)
            cm = confusion_matrix(y_test, y_pred_dnn)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap="Oranges", ax=ax)
            ax.set_title("DNN Confusion Matrix")
            return fig

# --------------------
# APP
# --------------------
app = App(app_ui, server)
