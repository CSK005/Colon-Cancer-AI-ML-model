# app.py - Shiny for Python: Colon Cancer Classifier (Alternative with HTML tables)

from shiny import App, render, ui, reactive
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import traceback

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from xgboost import XGBClassifier
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Set matplotlib backend for server deployment
plt.switch_backend('Agg')

# --------------------
# UI
# --------------------
app_ui = ui.page_fluid(
    ui.h2("Colon Cancer Classifier Models"),
    ui.p("Early Detection and Prediction using Machine Learning & Deep Learning"),

    ui.layout_sidebar(
        ui.sidebar(
            ui.input_file("train_file", "Upload Training CSV", multiple=True, accept=".csv"),
            ui.input_file("test_file", "Upload Testing CSV", multiple=True, accept=".csv"),
            ui.input_select("feature_select", "Select feature to visualize", choices=[], selected=None),
            ui.input_action_button("run_models", "Run Models", class_="btn-primary"),
            ui.br(),
            ui.output_text("status_text"),
        ),
        # Main content area
        ui.div(
            ui.h4("Training Data Preview"),
            ui.output_ui("train_preview"),
            ui.br(),
            ui.h4("Testing Data Preview"), 
            ui.output_ui("test_preview"),
            ui.br(),
            ui.h4("Feature Distribution"),
            ui.output_plot("feature_plot"),
            ui.br(),
            ui.h4("Model Results"),
            ui.output_text("rf_acc"),
            ui.output_plot("rf_cm"),
            ui.output_text("xgb_acc"), 
            ui.output_plot("xgb_cm"),
            ui.output_text("dnn_acc"),
            ui.output_plot("dnn_cm"),
        )
    ),

    ui.hr(),
    ui.h3("About"),
    ui.p("This Colon Cancer classification model was developed to provide insights "
         "into colon cancer prediction using genomic data."),
    ui.a("GitHub Repository", href="https://github.com/CSK005/Colon-Cancer-AI-ML-model", target="_blank"),
    ui.tags.ul(
        ui.tags.li("CHANDRASHEKAR K"),
        ui.tags.li("Dr. VIDYA NIRANJAN"),
        ui.tags.li("ANAGHA S SETLUR"),
    )
)

# --------------------
# Helper Functions
# --------------------
def df_to_html_table(df, max_rows=5):
    """Convert DataFrame to HTML table without using styling"""
    if df.empty:
        return ui.p("No data available")
    
    df_display = df.head(max_rows)
    
    # Create table headers
    headers = [ui.tags.th(col) for col in df_display.columns]
    header_row = ui.tags.tr(*headers)
    
    # Create table rows
    rows = []
    for _, row in df_display.iterrows():
        cells = [ui.tags.td(str(val)) for val in row]
        rows.append(ui.tags.tr(*cells))
    
    # Combine into table
    table = ui.tags.table(
        ui.tags.thead(header_row),
        ui.tags.tbody(*rows),
        class_="table table-striped table-sm"
    )
    
    return table

# --------------------
# Data Preprocessing
# --------------------
def preprocess(files):
    """Preprocess uploaded CSV files"""
    try:
        if not files:
            return None, None, "No files uploaded"

        dfs = []
        for f in files:
            try:
                df = pd.read_csv(f["datapath"])
                dfs.append(df)
            except Exception as e:
                return None, None, f"Error reading file {f['name']}: {str(e)}"

        if not dfs:
            return None, None, "No valid CSV files found"

        df = pd.concat(dfs, ignore_index=True)

        # Basic cleaning
        df = df.replace(".", np.nan)
        df = df.fillna(0)

        # Encode Chr column if present
        chr_map = {f'chr{i}': i for i in range(1, 23)}
        chr_map.update({'chrX': 23, 'chrY': 24})
        if 'Chr' in df.columns:
            df['Chr'] = df['Chr'].map(chr_map).fillna(0).astype(int)

        # Encode categorical columns (only if they exist)
        encode_cols = [
            "Func.refGene", "ExonicFunc.refGene", "Polyphen2_HDIV_pred",
            "Polyphen2_HVAR_pred", "SIFT_pred", "MutationTaster_pred",
            "MutationAssessor_pred", "CLNSIG"
        ]
        
        for col in encode_cols:
            if col in df.columns:
                try:
                    le = LabelEncoder()
                    df[col] = le.fit_transform(df[col].astype(str))
                except Exception as e:
                    print(f"Warning: Could not encode column {col}: {e}")

        # Normalize numeric columns (only if they exist)
        scale_cols = ["CADD", "CADD_Phred", "MutationTaster_score",
                      "MutationAssessor_score", "AF", "AF_popmax"]
        scale_cols = [c for c in scale_cols if c in df.columns]
        
        if scale_cols:
            try:
                scaler = MinMaxScaler()
                df[scale_cols] = scaler.fit_transform(df[scale_cols])
            except Exception as e:
                print(f"Warning: Could not scale columns: {e}")

        # Determine target column
        target_col = None
        if 'Func.refGene' in df.columns:
            target_col = 'Func.refGene'
        elif 'target' in df.columns:
            target_col = 'target'
        elif 'label' in df.columns:
            target_col = 'label'
        else:
            # Use the last column as target
            target_col = df.columns[-1]

        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Ensure all features are numeric
        for col in X.columns:
            try:
                X[col] = pd.to_numeric(X[col], errors='coerce')
            except:
                pass
        
        X = X.fillna(0)
        
        return X, y, "Success"

    except Exception as e:
        return None, None, f"Preprocessing error: {str(e)}"

# --------------------
# SERVER
# --------------------
def server(input, output, session):

    # Reactive values
    train_data = reactive.Value(None)
    test_data = reactive.Value(None)
    status_msg = reactive.Value("Ready to upload files")

    # Status output
    @output
    @render.text
    def status_text():
        return status_msg.get()

    # Load Data
    @reactive.Effect
    @reactive.event(input.train_file, input.test_file)
    def _load_data():
        try:
            status_msg.set("Loading data...")
            
            # Process training data
            if input.train_file():
                X_train, y_train, train_status = preprocess(input.train_file())
                if X_train is not None:
                    train_data.set((X_train, y_train))
                    status_msg.set(f"Training data loaded: {X_train.shape[0]} rows, {X_train.shape[1]} features")
                else:
                    status_msg.set(f"Training data error: {train_status}")
                    return
            
            # Process testing data
            if input.test_file():
                X_test, y_test, test_status = preprocess(input.test_file())
                if X_test is not None:
                    test_data.set((X_test, y_test))
                    
                    # Update feature choices for dropdown
                    if train_data.get():
                        feature_choices = list(train_data.get()[0].columns)
                        ui.update_select("feature_select", choices=feature_choices, selected=feature_choices[0] if feature_choices else None)
                    
                    status_msg.set("Both datasets loaded successfully. Ready to run models.")
                else:
                    status_msg.set(f"Testing data error: {test_status}")
            
        except Exception as e:
            status_msg.set(f"Error loading data: {str(e)}")

    # Preview Tables using HTML
    @output
    @render.ui
    def train_preview():
        try:
            if train_data.get():
                df = train_data.get()[0]
                return df_to_html_table(df, max_rows=5)
            return ui.p("No training data uploaded", class_="text-muted")
        except Exception as e:
            return ui.p(f"Error displaying data: {str(e)}", class_="text-danger")

    @output
    @render.ui  
    def test_preview():
        try:
            if test_data.get():
                df = test_data.get()[0]
                return df_to_html_table(df, max_rows=5)
            return ui.p("No testing data uploaded", class_="text-muted")
        except Exception as e:
            return ui.p(f"Error displaying data: {str(e)}", class_="text-danger")

    # Feature Distribution Plot
    @output
    @render.plot
    def feature_plot():
        try:
            if train_data.get():
                df = train_data.get()[0]
                feature = input.feature_select()
                if feature and feature in df.columns:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Handle different data types
                    data = df[feature].dropna()
                    if len(data) > 0:
                        if data.dtype in ['object', 'category']:
                            data.value_counts().plot(kind='bar', ax=ax)
                        else:
                            sns.histplot(data, bins=min(30, len(data.unique())), kde=True, ax=ax)
                        
                        ax.set_title(f"Distribution of {feature}")
                        ax.tick_params(axis='x', rotation=45)
                        plt.tight_layout()
                    return fig
        except Exception as e:
            fig, ax = plt.subplots()
            ax.text(0.5, 0.5, f"Error creating plot: {str(e)}", ha='center', va='center')
            return fig

    # Model outputs (initialized as empty)
    @output
    @render.text
    def rf_acc():
        return ""

    @output
    @render.plot
    def rf_cm():
        return plt.figure()

    @output
    @render.text
    def xgb_acc():
        return ""

    @output
    @render.plot
    def xgb_cm():
        return plt.figure()

    @output
    @render.text
    def dnn_acc():
        return ""

    @output
    @render.plot
    def dnn_cm():
        return plt.figure()

    # Run Models
    @reactive.Effect
    @reactive.event(input.run_models)
    def _train_models():
        try:
            if not (train_data.get() and test_data.get()):
                status_msg.set("Please upload both training and testing data first")
                return

            status_msg.set("Training models...")
            
            X_train, y_train = train_data.get()
            X_test, y_test = test_data.get()

            # Ensure consistent features
            common_features = X_train.columns.intersection(X_test.columns)
            X_train = X_train[common_features]
            X_test = X_test[common_features]

            if len(common_features) == 0:
                status_msg.set("No common features between training and testing data")
                return

            # ---- Random Forest ----
            try:
                status_msg.set("Training Random Forest...")
                rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
                rf.fit(X_train, y_train)
                y_pred_rf = rf.predict(X_test)
                acc_rf = accuracy_score(y_test, y_pred_rf)

                @output
                @render.text
                def rf_acc():
                    return f"Random Forest Accuracy: {acc_rf:.3f}"

                @output
                @render.plot
                def rf_cm():
                    try:
                        cm = confusion_matrix(y_test, y_pred_rf)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", ax=ax)
                        ax.set_title("Random Forest Confusion Matrix")
                        plt.tight_layout()
                        return fig
                    except Exception as e:
                        fig, ax = plt.subplots()
                        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                        return fig

            except Exception as e:
                status_msg.set(f"Random Forest error: {str(e)}")

            # ---- XGBoost ----
            try:
                status_msg.set("Training XGBoost...")
                xgb = XGBClassifier(n_estimators=50, random_state=42, eval_metric='mlogloss', verbosity=0)
                xgb.fit(X_train, y_train)
                y_pred_xgb = xgb.predict(X_test)
                acc_xgb = accuracy_score(y_test, y_pred_xgb)

                @output
                @render.text
                def xgb_acc():
                    return f"XGBoost Accuracy: {acc_xgb:.3f}"

                @output
                @render.plot
                def xgb_cm():
                    try:
                        cm = confusion_matrix(y_test, y_pred_xgb)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap="Greens", ax=ax)
                        ax.set_title("XGBoost Confusion Matrix")
                        plt.tight_layout()
                        return fig
                    except Exception as e:
                        fig, ax = plt.subplots()
                        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                        return fig

            except Exception as e:
                status_msg.set(f"XGBoost error: {str(e)}")

            # ---- Deep Neural Network ----
            try:
                status_msg.set("Training Deep Neural Network...")
                
                # Prepare data for DNN
                n_classes = len(np.unique(y_train))
                
                model = Sequential([
                    Dense(64, input_dim=X_train.shape[1], activation='relu'),
                    Dense(32, activation='relu'), 
                    Dense(n_classes, activation='softmax' if n_classes > 2 else 'sigmoid')
                ])
                
                loss_fn = 'sparse_categorical_crossentropy' if n_classes > 2 else 'binary_crossentropy'
                model.compile(loss=loss_fn, optimizer='adam', metrics=['accuracy'])
                
                # Train with reduced epochs for faster deployment
                model.fit(X_train, y_train, epochs=5, batch_size=32, verbose=0, validation_split=0.2)

                _, acc_dnn = model.evaluate(X_test, y_test, verbose=0)

                @output
                @render.text
                def dnn_acc():
                    return f"Deep Neural Network Accuracy: {acc_dnn:.3f}"

                @output
                @render.plot
                def dnn_cm():
                    try:
                        y_pred_prob = model.predict(X_test, verbose=0)
                        if n_classes > 2:
                            y_pred_dnn = np.argmax(y_pred_prob, axis=1)
                        else:
                            y_pred_dnn = (y_pred_prob > 0.5).astype(int).flatten()
                            
                        cm = confusion_matrix(y_test, y_pred_dnn)
                        fig, ax = plt.subplots(figsize=(8, 6))
                        sns.heatmap(cm, annot=True, fmt='d', cmap="Oranges", ax=ax)
                        ax.set_title("Deep Neural Network Confusion Matrix")
                        plt.tight_layout()
                        return fig
                    except Exception as e:
                        fig, ax = plt.subplots()
                        ax.text(0.5, 0.5, f"Error: {str(e)}", ha='center', va='center')
                        return fig

            except Exception as e:
                status_msg.set(f"DNN error: {str(e)}")

            status_msg.set("All models trained successfully!")

        except Exception as e:
            status_msg.set(f"Training error: {str(e)}")
            print(f"Full traceback: {traceback.format_exc()}")

# --------------------
# APP
# --------------------
app = App(app_ui, server)

if __name__ == "__main__":
    app.run()
