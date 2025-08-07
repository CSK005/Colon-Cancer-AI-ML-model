from shiny import App, ui, render, reactive
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# User Interface
app_ui = ui.page_fluid(
    ui.panel_title("Colon Cancer Classifier Models"),
    ui.markdown("""
    ### Early Detection and Prediction using Machine Learning & Deep Learning
    This application provides insights into colon cancer prediction using exome sequencing data.
    """),
    ui.h4("Example Dataset Format"),
    ui.output_data_frame("example_df"),
    ui.input_file("train_csv", "Upload training CSV files", multiple=True, accept=[".csv"]),
    ui.input_file("test_csv", "Upload test CSV files", multiple=True, accept=[".csv"]),
    ui.output_table("train_preview"),
    ui.output_table("test_preview"),
    ui.input_select("feature_plot", "Feature to visualize", choices=["CADD", "AF", "CADD_Phred", "MutationTaster_score", "MutationAssessor_score", "AF_popmax"]),
    ui.output_plot("feature_plot"),
    ui.h4("Random Forest Results"),
    ui.output_text_verbatim("rf_accuracy"),
    ui.output_plot("rf_confmat"),
    ui.output_plot("rf_class_report"),
    ui.h4("XGBoost Results"),
    ui.output_text_verbatim("xgb_accuracy"),
    ui.output_plot("xgb_confmat"),
    ui.output_plot("xgb_class_report"),
    ui.h4("DNN Results"),
    ui.output_text_verbatim("dnn_accuracy"),
    ui.output_plot("dnn_confmat"),
    ui.output_plot("dnn_class_report"),
    ui.output_table("model_summary"),
    ui.markdown("---"),
    ui.h4("About Us"),
    ui.markdown("""
    This Colon Cancer classification model was developed to provide insights into colon cancer prediction using exome sequencing data,
    aiding in personalized treatment plans and early colon cancer diagnosis.

    Developed by:
    - CHANDRASHEKAR K
    - Dr. VIDYA NIRANJAN
    - ANAGHA S SETLUR
    (Department of Biotechnology, RV College of Engineering)
    """),
    ui.markdown("For model source code, visit: [GitHub Repository](https://github.com/CSK005/Colon-Cancer-AI-ML-model/blob/main/app.py)")
)

# Example dataframe
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

def preprocess_data(files):
    if not files:
        return None, None
    try:
        dfs = [pd.read_csv(f['datapath']) for f in files]
        df = pd.concat(dfs, ignore_index=True)
    except Exception as e:
        return None, None
    df.replace(".", np.nan, inplace=True)
    df.fillna(0, inplace=True)
    chr_map = {f'chr{i}': i for i in range(1, 23)}
    chr_map.update({'chrX': 23, 'chrY': 24})
    df['Chr'] = df['Chr'].map(chr_map).fillna(0).astype(int)
    encode_cols = ["Func.refGene", "ExonicFunc.refGene", "Polyphen2_HDIV_pred", "Polyphen2_HVAR_pred", "SIFT_pred", "MutationTaster_pred", "MutationAssessor_pred", "CLNSIG"]
    for col in encode_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    obj_cols = df.select_dtypes(include=['object']).columns
    for col in obj_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    scale_cols = [c for c in ["CADD", "CADD_Phred", "MutationTaster_score", "MutationAssessor_score", "AF", "AF_popmax"] if c in df.columns]
    if scale_cols:
        scaler = MinMaxScaler()
        df[scale_cols] = scaler.fit_transform(df[scale_cols])
    if 'Func.refGene' not in df.columns:
        return None, None
    X = df.drop(columns=['Func.refGene'])
    y = df['Func.refGene']
    return X, y

def server(input, output, session):
    @render.data_frame
    def example_df_out():
        return example_df

    @reactive.calc
    def train_data():
        files = input.train_csv()
        if not files:
            return None, None
        return preprocess_data(files)

    @reactive.calc
    def test_data():
        files = input.test_csv()
        if not files:
            return None, None
        return preprocess_data(files)

    @render.table
    def train_preview():
        X, _ = train_data()
        return X.head() if X is not None else pd.DataFrame()

    @render.table
    def test_preview():
        X, _ = test_data()
        return X.head() if X is not None else pd.DataFrame()

    @render.plot
    def feature_plot():
        X, _ = train_data()
        if X is None:
            return
        feat = input.feature_plot()
        fig, ax = plt.subplots()
        sns.histplot(X[feat], bins=30, kde=True, ax=ax)
        return fig

    # ML and Results
    # Use more robust reactive logic and error handling in production code
    @reactive.calc
    def model_results():
        X_train, y_train = train_data()
        X_test, y_test = test_data()
        if X_train is None or X_test is None or y_train is None or y_test is None:
            return None
        results = {}
        try:
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train)
            y_pred_rf = rf.predict(X_test)
            acc_rf = accuracy_score(y_test, y_pred_rf)
            results['rf'] = (acc_rf, confusion_matrix(y_test, y_pred_rf), classification_report(y_test, y_pred_rf, output_dict=True))
        except Exception as e:
            results['rf'] = None
        try:
            xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', n_estimators=100, random_state=42)
            xgb.fit(X_train, y_train)
            y_pred_xgb = xgb.predict(X_test)
            acc_xgb = accuracy_score(y_test, y_pred_xgb)
            results['xgb'] = (acc_xgb, confusion_matrix(y_test, y_pred_xgb), classification_report(y_test, y_pred_xgb, output_dict=True))
        except Exception as e:
            results['xgb'] = None
        try:
            model = Sequential([
                Dense(128, input_dim=X_train.shape[1], activation='relu'),
                Dense(64, activation='relu'),
                Dense(32, activation='relu'),
                Dense(len(np.unique(y_train)), activation='softmax')
            ])
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
            model.fit(X_train.values, y_train.values, epochs=15, batch_size=32, verbose=0)
            loss, acc_dnn = model.evaluate(X_test.values, y_test.values, verbose=0)
            y_pred_dnn_prob = model.predict(X_test.values)
            y_pred_dnn = np.argmax(y_pred_dnn_prob, axis=1)
            results['dnn'] = (acc_dnn, confusion_matrix(y_test, y_pred_dnn), classification_report(y_test, y_pred_dnn, output_dict=True))
        except Exception as e:
            results['dnn'] = None
        return results

    # Output blocks (update for each model as above)
    @render.text
    def rf_accuracy():
        res = model_results()
        if res and res.get('rf'):
            return f"Accuracy: {res['rf'][0]:.2f}"
        return "No data."

    @render.plot
    def rf_confmat():
        res = model_results()
        if res and res.get('rf'):
            cm = res['rf'][1]
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            return fig
        return

    @render.plot
    def rf_class_report():
        res = model_results()
        if res and res.get('rf'):
            rep = pd.DataFrame(res['rf'][2]).transpose().iloc[:-3, :3]
            fig, ax = plt.subplots()
            rep.plot(kind='bar', ax=ax)
            ax.set_ylim(0, 1)
            return fig
        return

    # Repeat output logic for XGBoost and DNN results...
    # For brevity, show only accuracy output for remaining models:
    @render.text
    def xgb_accuracy():
        res = model_results()
        if res and res.get('xgb'):
            return f"Accuracy: {res['xgb'][0]:.2f}"
        return "No data."

    @render.text
    def dnn_accuracy():
        res = model_results()
        if res and res.get('dnn'):
            return f"Accuracy: {res['dnn'][0]:.2f}"
        return "No data."

    # Similarly add confmat/class_report blocks for XGBoost and DNN...

    @render.table
    def model_summary():
        res = model_results()
        models = ["Random Forest", "XGBoost", "DNN"]
        accs = []
        for m in ["rf", "xgb", "dnn"]:
            accs.append(res[m][0] if res and res.get(m) else None)
        df = pd.DataFrame({"Model": models, "Accuracy": accs})
        return df

app = App(app_ui, server)
