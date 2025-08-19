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
    ui.div(
        ui.h1("Colon Cancer Classifier Models", class_="text-center text-primary"),
        ui.p("Early Detection and Prediction using Machine Learning & Deep Learning", 
             class_="lead text-center text-muted"),
        class_="mb-4"
    ),
    
    # Information Section
    ui.nav_panel(
        ui.navset_tab(
            ui.nav_panel(
                "📊 Upload & Analyze",
                ui.layout_sidebar(
                    ui.sidebar(
                        ui.h5("📁 Data Upload", class_="text-primary"),
                        ui.input_file("train_file", "Upload Training CSV", multiple=True, accept=".csv"),
                        ui.input_file("test_file", "Upload Testing CSV", multiple=True, accept=".csv"),
                        ui.br(),
                        ui.h5("🔍 Visualization", class_="text-primary"),
                        ui.input_select("feature_select", "Select feature to visualize", choices=[], selected=None),
                        ui.br(),
                        ui.h5("🚀 Model Training", class_="text-primary"),
                        ui.input_action_button("run_models", "Run All Models", class_="btn-primary btn-lg"),
                        ui.br(),
                        ui.br(),
                        ui.div(
                            ui.h6("📈 Status", class_="text-info"),
                            ui.output_text("status_text"),
                            class_="alert alert-info"
                        ),
                        width="300px"
                    ),
                    # Main content area
                    ui.div(
                        ui.h4("📋 Training Data Preview"),
                        ui.output_ui("train_preview"),
                        ui.br(),
                        ui.h4("📋 Testing Data Preview"), 
                        ui.output_ui("test_preview"),
                        ui.br(),
                        ui.h4("📊 Feature Distribution"),
                        ui.output_plot("feature_plot"),
                        ui.br(),
                        ui.h4("🎯 Model Performance Results"),
                        ui.div(
                            ui.div(
                                ui.h5("🌳 Random Forest"),
                                ui.output_text("rf_acc"),
                                ui.output_plot("rf_cm"),
                                class_="col-md-4"
                            ),
                            ui.div(
                                ui.h5("⚡ XGBoost"),
                                ui.output_text("xgb_acc"),
                                ui.output_plot("xgb_cm"),
                                class_="col-md-4"
                            ),
                            ui.div(
                                ui.h5("🧠 Deep Neural Network"),
                                ui.output_text("dnn_acc"),
                                ui.output_plot("dnn_cm"),
                                class_="col-md-4"
                            ),
                            class_="row"
                        )
                    )
                )
            ),
            
            ui.nav_panel(
                "📖 How to Use",
                ui.div(
                    ui.h3("🚀 Getting Started Guide"),
                    
                    ui.div(
                        ui.h4("📂 Step 1: Prepare Your Data"),
                        ui.p("Your CSV files should contain genomic variant data with the following structure:"),
                        
                        ui.h5("📊 Sample Training Data Format:"),
                        ui.tags.div(
                            ui.tags.table(
                                ui.tags.thead(
                                    ui.tags.tr(
                                        ui.tags.th("Chr"), ui.tags.th("Start"), ui.tags.th("End"), 
                                        ui.tags.th("Func.refGene"), ui.tags.th("CADD"), ui.tags.th("SIFT_pred"),
                                        ui.tags.th("Polyphen2_HDIV_pred"), ui.tags.th("AF")
                                    )
                                ),
                                ui.tags.tbody(
                                    ui.tags.tr(
                                        ui.tags.td("chr1"), ui.tags.td("12345"), ui.tags.td("12346"),
                                        ui.tags.td("exonic"), ui.tags.td("15.2"), ui.tags.td("T"),
                                        ui.tags.td("B"), ui.tags.td("0.001")
                                    ),
                                    ui.tags.tr(
                                        ui.tags.td("chr2"), ui.tags.td("67890"), ui.tags.td("67891"),
                                        ui.tags.td("intronic"), ui.tags.td("8.7"), ui.tags.td("D"),
                                        ui.tags.td("P"), ui.tags.td("0.005")
                                    ),
                                    ui.tags.tr(
                                        ui.tags.td("chr3"), ui.tags.td("11111"), ui.tags.td("11112"),
                                        ui.tags.td("UTR3"), ui.tags.td("12.1"), ui.tags.td("T"),
                                        ui.tags.td("B"), ui.tags.td("0.002")
                                    )
                                ),
                                class_="table table-striped table-sm"
                            ),
                            class_="table-responsive"
                        ),
                        
                        ui.h5("📊 Sample Testing Data Format:"),
                        ui.tags.div(
                            ui.tags.table(
                                ui.tags.thead(
                                    ui.tags.tr(
                                        ui.tags.th("Chr"), ui.tags.th("Start"), ui.tags.th("End"), 
                                        ui.tags.th("Func.refGene"), ui.tags.th("CADD"), ui.tags.th("SIFT_pred"),
                                        ui.tags.th("Polyphen2_HDIV_pred"), ui.tags.th("AF")
                                    )
                                ),
                                ui.tags.tbody(
                                    ui.tags.tr(
                                        ui.tags.td("chr4"), ui.tags.td("22222"), ui.tags.td("22223"),
                                        ui.tags.td("exonic"), ui.tags.td("18.5"), ui.tags.td("D"),
                                        ui.tags.td("P"), ui.tags.td("0.003")
                                    ),
                                    ui.tags.tr(
                                        ui.tags.td("chr5"), ui.tags.td("33333"), ui.tags.td("33334"),
                                        ui.tags.td("splicing"), ui.tags.td("25.1"), ui.tags.td("D"),
                                        ui.tags.td("D"), ui.tags.td("0.001")
                                    )
                                ),
                                class_="table table-striped table-sm"
                            ),
                            class_="table-responsive"
                        ),
                        
                        ui.div(
                            ui.h5("📋 Required Columns:"),
                            ui.tags.ul(
                                ui.tags.li(ui.tags.strong("Func.refGene"), " (Target): Functional annotation (exonic, intronic, splicing, etc.)"),
                                ui.tags.li(ui.tags.strong("Chr"), " (Optional): Chromosome identifier (chr1, chr2, etc.)"),
                                ui.tags.li(ui.tags.strong("CADD, CADD_Phred"), " (Optional): Pathogenicity scores"),
                                ui.tags.li(ui.tags.strong("SIFT_pred, Polyphen2_*_pred"), " (Optional): Prediction scores"),
                                ui.tags.li(ui.tags.strong("AF, AF_popmax"), " (Optional): Allele frequencies"),
                                ui.tags.li(ui.tags.strong("MutationTaster_*, MutationAssessor_*"), " (Optional): Additional scores")
                            ),
                            class_="alert alert-info"
                        ),
                        class_="mb-4"
                    ),
                    
                    ui.div(
                        ui.h4("⚙️ Step 2: Upload and Process"),
                        ui.tags.ol(
                            ui.tags.li("Upload your training CSV file(s) using the 'Upload Training CSV' button"),
                            ui.tags.li("Upload your testing CSV file(s) using the 'Upload Testing CSV' button"),
                            ui.tags.li("Preview your data in the 'Upload & Analyze' tab"),
                            ui.tags.li("Select a feature to visualize its distribution"),
                            ui.tags.li("Click 'Run All Models' to start the analysis")
                        ),
                        class_="mb-4"
                    ),
                    
                    ui.div(
                        ui.h4("📊 Step 3: Interpret Results"),
                        ui.p("The app will generate:"),
                        ui.tags.ul(
                            ui.tags.li("Accuracy scores for each model"),
                            ui.tags.li("Confusion matrices showing prediction performance"),
                            ui.tags.li("Feature distribution plots for data exploration")
                        ),
                        class_="mb-4"
                    ),
                    
                    class_="container-fluid"
                )
            ),
            
            ui.nav_panel(
                "🤖 Models Info",
                ui.div(
                    ui.h3("🧬 Machine Learning Models for Colon Cancer Classification"),
                    
                    ui.div(
                        ui.div(
                            ui.div(
                                ui.h4("🌳 Random Forest Classifier"),
                                ui.p("An ensemble learning method that constructs multiple decision trees and merges them together to get more accurate and stable predictions."),
                                ui.tags.strong("Key Features:"),
                                ui.tags.ul(
                                    ui.tags.li("Handles missing values well"),
                                    ui.tags.li("Provides feature importance rankings"),
                                    ui.tags.li("Reduces overfitting compared to single decision trees"),
                                    ui.tags.li("Works well with genomic data")
                                ),
                                ui.tags.strong("Best for: "), "Understanding which genetic features are most important for classification",
                                class_="card-body"
                            ),
                            class_="card mb-3"
                        ),
                        
                        ui.div(
                            ui.div(
                                ui.h4("⚡ XGBoost (Extreme Gradient Boosting)"),
                                ui.p("A powerful gradient boosting framework that uses ensemble of weak learners (decision trees) to create a strong predictor."),
                                ui.tags.strong("Key Features:"),
                                ui.tags.ul(
                                    ui.tags.li("High performance and speed"),
                                    ui.tags.li("Handles missing values automatically"), 
                                    ui.tags.li("Built-in regularization to prevent overfitting"),
                                    ui.tags.li("Excellent for structured/tabular data")
                                ),
                                ui.tags.strong("Best for: "), "Achieving high accuracy on genomic variant classification tasks",
                                class_="card-body"
                            ),
                            class_="card mb-3"
                        ),
                        
                        ui.div(
                            ui.div(
                                ui.h4("🧠 Deep Neural Network (DNN)"),
                                ui.p("A multi-layer neural network that can learn complex non-linear patterns in genomic data through backpropagation."),
                                ui.tags.strong("Architecture:"),
                                ui.tags.ul(
                                    ui.tags.li("Input Layer: Matches number of genomic features"),
                                    ui.tags.li("Hidden Layer 1: 64 neurons with ReLU activation"),
                                    ui.tags.li("Hidden Layer 2: 32 neurons with ReLU activation"),
                                    ui.tags.li("Output Layer: Softmax for multi-class classification")
                                ),
                                ui.tags.strong("Best for: "), "Capturing complex interactions between genetic variants",
                                class_="card-body"
                            ),
                            class_="card mb-3"
                        ),
                        class_="row"
                    ),
                    
                    ui.div(
                        ui.h4("🎯 Model Comparison & Selection"),
                        ui.tags.table(
                            ui.tags.thead(
                                ui.tags.tr(
                                    ui.tags.th("Model"), ui.tags.th("Strengths"), ui.tags.th("Use Case"), ui.tags.th("Interpretability")
                                )
                            ),
                            ui.tags.tbody(
                                ui.tags.tr(
                                    ui.tags.td("Random Forest"), 
                                    ui.tags.td("Feature importance, handles missing data"), 
                                    ui.tags.td("Exploratory analysis"), 
                                    ui.tags.td("High")
                                ),
                                ui.tags.tr(
                                    ui.tags.td("XGBoost"), 
                                    ui.tags.td("High accuracy, fast training"), 
                                    ui.tags.td("Production deployment"), 
                                    ui.tags.td("Medium")
                                ),
                                ui.tags.tr(
                                    ui.tags.td("Deep Neural Network"), 
                                    ui.tags.td("Complex pattern recognition"), 
                                    ui.tags.td("Large datasets"), 
                                    ui.tags.td("Low")
                                )
                            ),
                            class_="table table-striped"
                        ),
                        class_="mt-4"
                    ),
                    
                    class_="container-fluid"
                )
            ),
            
            ui.nav_panel(
                "ℹ️ About",
                ui.div(
                    ui.h3("🧬 Colon Cancer AI/ML Research Platform"),
                    
                    ui.div(
                        ui.h4("🎯 Research Objective"),
                        ui.p("This platform was developed to advance colon cancer research by providing machine learning tools for analyzing genomic variant data. It helps researchers classify genetic variants and understand their potential impact on colon cancer development."),
                        class_="mb-4"
                    ),
                    
                    ui.div(
                        ui.h4("📊 Data Science Approach"),
                        ui.p("Our approach combines traditional machine learning with deep learning to provide comprehensive analysis:"),
                        ui.tags.ul(
                            ui.tags.li("📈 ", ui.tags.strong("Preprocessing:"), " Automated data cleaning, encoding, and normalization"),
                            ui.tags.li("🔍 ", ui.tags.strong("Feature Engineering:"), " Chromosome mapping and categorical encoding"),
                            ui.tags.li("🤖 ", ui.tags.strong("Multi-Model Ensemble:"), " Three complementary algorithms for robust predictions"),
                            ui.tags.li("📊 ", ui.tags.strong("Visualization:"), " Interactive plots and confusion matrices for result interpretation")
                        ),
                        class_="mb-4"
                    ),
                    
                    ui.div(
                        ui.h4("🔬 For Researchers"),
                        ui.div(
                            ui.h5("📚 Publication & Citation"),
                            ui.p("If you use this tool in your research, please cite our work:"),
                            ui.tags.blockquote(
                                "Colon Cancer Classification using Machine Learning and Deep Learning Approaches on Genomic Variant Data",
                                class_="blockquote text-muted"
                            ),
                            ui.a("📖 GitHub Repository", href="https://github.com/CSK005/Colon-Cancer-AI-ML-model", 
                                 target="_blank", class_="btn btn-outline-primary"),
                        ),
                        class_="mb-4"
                    ),
                    
                    ui.div(
                        ui.h4("👥 Research Team"),
                        ui.div(
                            ui.div(
                                ui.h5("👨‍💻 CHANDRASHEKAR K"),
                                ui.p("Lead Developer & Data Scientist", class_="text-muted"),
                                class_="col-md-4"
                            ),
                            ui.div(
                                ui.h5("👩‍🔬 Dr. VIDYA NIRANJAN"),
                                ui.p("Principal Investigator & Research Supervisor", class_="text-muted"),
                                class_="col-md-4"
                            ),
                            ui.div(
                                ui.h5("👩‍💻 ANAGHA S SETLUR"),
                                ui.p("Research Collaborator", class_="text-muted"),
                                class_="col-md-4"
                            ),
                            class_="row"
                        ),
                        class_="mb-4"
                    ),
                    
                    ui.div(
                        ui.h4("📧 Contact & Support"),
                        ui.p("For research collaborations, technical support, or questions about the methodology, please reach out through our GitHub repository or institutional contacts."),
                        ui.div(
                            ui.a("🐛 Report Issues", href="https://github.com/CSK005/Colon-Cancer-AI-ML-model/issues", 
                                 target="_blank", class_="btn btn-outline-secondary me-2"),
                            ui.a("💡 Feature Requests", href="https://github.com/CSK005/Colon-Cancer-AI-ML-model/discussions", 
                                 target="_blank", class_="btn btn-outline-info"),
                        ),
                        class_="alert alert-light"
                    ),
                    
                    class_="container-fluid"
                )
            ),
            
            id="main_tabs"
        )
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
