import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from xgboost import XGBClassifier
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
import shap
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import io
import base64
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title="Advanced Cancer Classification Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .stAlert {
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title and Introduction
st.markdown('<h1 class="main-header">🧬 Advanced Cancer Classification Platform</h1>', unsafe_allow_html=True)

st.markdown("""
### AI-Powered Cancer Detection & Risk Assessment
This enhanced platform leverages state-of-the-art machine learning and deep learning models for precise cancer classification 
using genomic data. Features include ensemble learning, model interpretability, and clinical decision support.
""")

# Sidebar Configuration
st.sidebar.title("⚙️ Configuration Panel")
st.sidebar.markdown("---")

# Model Selection
st.sidebar.subheader("Model Selection")
available_models = {
    "Random Forest": True,
    "XGBoost": True,
    "LightGBM": True,
    "Deep Neural Network": True,
    "Ensemble (Voting)": True
}

selected_models = {}
for model_name, default in available_models.items():
    selected_models[model_name] = st.sidebar.checkbox(model_name, value=default)

# Advanced Settings
st.sidebar.subheader("Advanced Settings")
use_feature_selection = st.sidebar.checkbox("Enable Feature Selection", value=True)
use_hyperparameter_tuning = st.sidebar.checkbox("Enable Hyperparameter Tuning", value=False)
use_cross_validation = st.sidebar.checkbox("Enable Cross-Validation", value=True)
show_interpretability = st.sidebar.checkbox("Show Model Interpretability", value=True)

# Data Upload Section
st.subheader("📁 Data Upload")
col1, col2 = st.columns(2)

with col1:
    st.markdown("**Training Data**")
    uploaded_files = st.file_uploader("Upload training CSV files", type=["csv"], accept_multiple_files=True, key="train")

with col2:
    st.markdown("**Testing Data**")
    uploaded_test_files = st.file_uploader("Upload testing CSV files", type=["csv"], accept_multiple_files=True, key="test")

# Enhanced preprocessing function
@st.cache_data
def enhanced_preprocess_data(files, is_training=True):
    """Enhanced data preprocessing with better error handling and feature engineering"""
    if not files:
        return None, None, None
    
    try:
        # Load and combine files
        dfs = []
        for file in files:
            df = pd.read_csv(file)
            dfs.append(df)
        df = pd.concat(dfs, ignore_index=True)
        
        # Store original shape
        original_shape = df.shape
        
        # Replace '.' with NaN and handle missing values
        df.replace(".", np.nan, inplace=True)
        
        # Advanced missing value handling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object']).columns
        
        # Fill numeric columns with median
        for col in numeric_cols:
            df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        for col in categorical_cols:
            df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown', inplace=True)
        
        # Convert chromosome names to numeric
        if 'Chr' in df.columns:
            chr_map = {f'chr{i}': i for i in range(1, 23)}
            chr_map.update({'chrX': 23, 'chrY': 24})
            df['Chr'] = df['Chr'].map(chr_map).fillna(0).astype(int)
        
        # Enhanced label encoding
        encode_cols = ["Func.refGene", "ExonicFunc.refGene", "Polyphen2_HDIV_pred", "Polyphen2_HVAR_pred",
                      "SIFT_pred", "MutationTaster_pred", "MutationAssessor_pred", "CLNSIG"]
        
        label_encoders = {}
        for col in encode_cols:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
        
        # Convert remaining object columns to numeric
        obj_cols = df.select_dtypes(include=['object']).columns
        for col in obj_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
        
        # Feature engineering
        if 'CADD' in df.columns and 'CADD_Phred' in df.columns:
            df['CADD_ratio'] = df['CADD'] / (df['CADD_Phred'] + 1e-6)
        
        if 'AF' in df.columns and 'AF_popmax' in df.columns:
            df['AF_difference'] = df['AF_popmax'] - df['AF']
        
        # Normalize numerical columns
        scale_cols = ["CADD", "CADD_Phred", "MutationTaster_score", "MutationAssessor_score", "AF", "AF_popmax"]
        scale_cols = [col for col in scale_cols if col in df.columns]
        
        if scale_cols:
            scaler = StandardScaler()
            df[scale_cols] = scaler.fit_transform(df[scale_cols])
        
        # Check for target column
        if 'Func.refGene' not in df.columns:
            st.error("Target column 'Func.refGene' is missing.")
            return None, None, None
        
        # Separate features and target
        X = df.drop(columns=['Func.refGene'])
        y = df['Func.refGene']
        
        # Create preprocessing info
        preprocessing_info = {
            'original_shape': original_shape,
            'processed_shape': df.shape,
            'feature_columns': X.columns.tolist(),
            'target_classes': y.unique().tolist(),
            'missing_values_handled': True
        }
        
        return X, y, preprocessing_info
        
    except Exception as e:
        st.error(f"Error processing data: {str(e)}")
        return None, None, None

# Enhanced model training functions
@st.cache_resource
def train_enhanced_models(X_train, y_train, X_test, y_test, selected_models, use_tuning=False):
    """Train multiple enhanced models with optional hyperparameter tuning"""
    models = {}
    results = {}
    
    if selected_models.get("Random Forest", False):
        if use_tuning:
            rf_params = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            }
            rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=3, scoring='accuracy')
            rf_grid.fit(X_train, y_train)
            rf_model = rf_grid.best_estimator_
        else:
            rf_model = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
            rf_model.fit(X_train, y_train)
        
        models['Random Forest'] = rf_model
        results['Random Forest'] = evaluate_model(rf_model, X_test, y_test)
    
    if selected_models.get("XGBoost", False):
        if use_tuning:
            xgb_params = {
                'n_estimators': [100, 200],
                'max_depth': [3, 6, 10],
                'learning_rate': [0.01, 0.1, 0.2]
            }
            xgb_grid = GridSearchCV(XGBClassifier(random_state=42, eval_metric='mlogloss'), xgb_params, cv=3)
            xgb_grid.fit(X_train, y_train)
            xgb_model = xgb_grid.best_estimator_
        else:
            xgb_model = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, eval_metric='mlogloss')
            xgb_model.fit(X_train, y_train)
        
        models['XGBoost'] = xgb_model
        results['XGBoost'] = evaluate_model(xgb_model, X_test, y_test)
    
    if selected_models.get("LightGBM", False):
        lgb_model = lgb.LGBMClassifier(n_estimators=200, random_state=42, verbose=-1)
        lgb_model.fit(X_train, y_train)
        models['LightGBM'] = lgb_model
        results['LightGBM'] = evaluate_model(lgb_model, X_test, y_test)
    
    if selected_models.get("Deep Neural Network", False):
        dnn_model = create_enhanced_dnn(X_train, y_train, X_test, y_test)
        models['Deep Neural Network'] = dnn_model
        results['Deep Neural Network'] = evaluate_dnn_model(dnn_model, X_test, y_test)
    
    if selected_models.get("Ensemble (Voting)", False) and len(models) > 1:
        estimators = [(name, model) for name, model in models.items() if name != 'Deep Neural Network']
        if estimators:
            ensemble = VotingClassifier(estimators=estimators, voting='soft')
            ensemble.fit(X_train, y_train)
            models['Ensemble (Voting)'] = ensemble
            results['Ensemble (Voting)'] = evaluate_model(ensemble, X_test, y_test)
    
    return models, results

def create_enhanced_dnn(X_train, y_train, X_test, y_test):
    """Create enhanced deep neural network with regularization"""
    n_features = X_train.shape[1]
    n_classes = len(np.unique(y_train))
    
    model = Sequential([
        Dense(256, input_dim=n_features, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[early_stopping],
        verbose=0
    )
    
    return model

def evaluate_model(model, X_test, y_test):
    """Comprehensive model evaluation"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return results

def evaluate_dnn_model(model, X_test, y_test):
    """Evaluate deep neural network model"""
    y_pred_proba = model.predict(X_test)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    results = {
        'accuracy': accuracy_score(y_test, y_pred),
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'classification_report': classification_report(y_test, y_pred, output_dict=True)
    }
    
    return results

# Feature selection function
def perform_feature_selection(X_train, y_train, X_test, n_features=20):
    """Perform feature selection using SelectKBest"""
    selector = SelectKBest(score_func=f_classif, k=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    return X_train_selected, X_test_selected, selected_features

# Visualization functions
def create_enhanced_visualizations(models, results, X_test, y_test):
    """Create comprehensive visualizations"""
    
    # Model comparison
    st.subheader("📊 Model Performance Comparison")
    
    # Accuracy comparison
    accuracies = {name: result['accuracy'] for name, result in results.items()}
    
    fig_acc = px.bar(
        x=list(accuracies.keys()),
        y=list(accuracies.values()),
        title="Model Accuracy Comparison",
        labels={'x': 'Model', 'y': 'Accuracy'},
        color=list(accuracies.values()),
        color_continuous_scale='viridis'
    )
    fig_acc.update_layout(showlegend=False)
    st.plotly_chart(fig_acc, use_container_width=True)
    
    # Confusion matrices
    st.subheader("🔍 Confusion Matrices")
    
    n_models = len(results)
    cols = st.columns(min(3, n_models))
    
    for idx, (name, result) in enumerate(results.items()):
        with cols[idx % 3]:
            fig, ax = plt.subplots(figsize=(6, 4))
            sns.heatmap(result['confusion_matrix'], annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title(f"{name}")
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
            plt.close()
    
    # Classification metrics
    st.subheader("📈 Detailed Classification Metrics")
    
    for name, result in results.items():
        with st.expander(f"{name} - Detailed Metrics"):
            report_df = pd.DataFrame(result['classification_report']).transpose()
            st.dataframe(report_df.round(3))

# SHAP interpretability function
def show_model_interpretability(model, X_test, feature_names):
    """Display SHAP explanations for model interpretability"""
    try:
        if hasattr(model, 'predict_proba'):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test.iloc[:100])  # Limit to first 100 samples
            
            # Summary plot
            st.subheader("🔬 Model Interpretability (SHAP)")
            
            fig, ax = plt.subplots(figsize=(10, 6))
            if isinstance(shap_values, list):
                shap.summary_plot(shap_values[0], X_test.iloc[:100], feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X_test.iloc[:100], feature_names=feature_names, show=False)
            st.pyplot(fig)
            plt.close()
            
    except Exception as e:
        st.warning(f"Could not generate SHAP explanations: {str(e)}")

# Main execution
if uploaded_files and uploaded_test_files:
    # Process data
    with st.spinner("Processing data..."):
        df_train_X, df_train_y, train_info = enhanced_preprocess_data(uploaded_files, is_training=True)
        df_test_X, df_test_y, test_info = enhanced_preprocess_data(uploaded_test_files, is_training=False)
    
    if df_train_X is not None and df_test_X is not None:
        # Data overview
        st.subheader("📋 Data Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Training Samples", len(df_train_X))
        with col2:
            st.metric("Testing Samples", len(df_test_X))
        with col3:
            st.metric("Features", len(df_train_X.columns))
        with col4:
            st.metric("Classes", len(np.unique(df_train_y)))
        
        # Feature selection
        if use_feature_selection:
            with st.spinner("Performing feature selection..."):
                n_features = st.slider("Number of features to select", 10, min(50, len(df_train_X.columns)), 20)
                df_train_X_selected, df_test_X_selected, selected_features = perform_feature_selection(
                    df_train_X, df_train_y, df_test_X, n_features
                )
                st.success(f"Selected {len(selected_features)} most important features")
                
                # Convert back to DataFrame for consistency
                df_train_X = pd.DataFrame(df_train_X_selected, columns=selected_features)
                df_test_X = pd.DataFrame(df_test_X_selected, columns=selected_features)
        
        # Model training
        with st.spinner("Training models..."):
            models, results = train_enhanced_models(
                df_train_X, df_train_y, df_test_X, df_test_y, 
                selected_models, use_hyperparameter_tuning
            )
        
        if models:
            # Display results
            create_enhanced_visualizations(models, results, df_test_X, df_test_y)
            
            # Model interpretability
            if show_interpretability and models:
                best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
                best_model = models[best_model_name]
                
                st.subheader(f"🔍 Model Interpretability - {best_model_name}")
                show_model_interpretability(best_model, df_test_X, df_test_X.columns)
            
            # Cross-validation results
            if use_cross_validation:
                st.subheader("✅ Cross-Validation Results")
                cv_results = {}
                
                for name, model in models.items():
                    if name != 'Deep Neural Network':  # Skip DNN for CV due to computational cost
                        cv_scores = cross_val_score(model, df_train_X, df_train_y, cv=5, scoring='accuracy')
                        cv_results[name] = {
                            'mean': cv_scores.mean(),
                            'std': cv_scores.std(),
                            'scores': cv_scores
                        }
                
                cv_df = pd.DataFrame({
                    'Model': list(cv_results.keys()),
                    'CV Mean Accuracy': [cv_results[name]['mean'] for name in cv_results.keys()],
                    'CV Std': [cv_results[name]['std'] for name in cv_results.keys()]
                })
                
                st.dataframe(cv_df.round(4))
            
            # Model download
            st.subheader("💾 Model Export")
            
            if st.button("Download Best Model"):
                best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
                best_model = models[best_model_name]
                
                # Save model
                model_buffer = io.BytesIO()
                joblib.dump(best_model, model_buffer)
                model_buffer.seek(0)
                
                st.download_button(
                    label=f"Download {best_model_name} Model",
                    data=model_buffer.getvalue(),
                    file_name=f"{best_model_name.lower().replace(' ', '_')}_model.pkl",
                    mime="application/octet-stream"
                )
        
        # Single sample prediction
        st.subheader("🔮 Single Sample Prediction")
        
        if models:
            best_model_name = max(results.keys(), key=lambda x: results[x]['accuracy'])
            best_model = models[best_model_name]
            
            with st.expander("Predict Single Sample"):
                sample_input = {}
                cols = st.columns(3)
                
                for idx, feature in enumerate(df_train_X.columns):
                    with cols[idx % 3]:
                        sample_input[feature] = st.number_input(
                            f"{feature}", 
                            value=float(df_train_X[feature].mean()),
                            format="%.6f"
                        )
                
                if st.button("Predict Sample"):
                    sample_df = pd.DataFrame([sample_input])
                    prediction = best_model.predict(sample_df)[0]
                    
                    if hasattr(best_model, 'predict_proba'):
                        probability = best_model.predict_proba(sample_df)[0]
                        max_prob = np.max(probability)
                        
                        st.success(f"Prediction: **{prediction}**")
                        st.info(f"Confidence: **{max_prob:.2%}**")
                        
                        # Show all class probabilities
                        prob_df = pd.DataFrame({
                            'Class': range(len(probability)),
                            'Probability': probability
                        })
                        st.bar_chart(prob_df.set_index('Class'))
                    else:
                        st.success(f"Prediction: **{prediction}**")

else:
    st.info("Please upload both training and testing CSV files to begin analysis.")
    
    # Show example data format
    st.subheader("📋 Expected Data Format")
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
    st.dataframe(example_df)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <h3>🧬 Advanced Cancer Classification Platform</h3>
    <p>Developed by: CHANDRASHEKAR K, Dr. VIDYA NIRANJAN, ANAGHA S SETLUR</p>
    <p>Department of Biotechnology, RV College of Engineering</p>
    <p><em>Empowering precision medicine through AI</em></p>
</div>
""", unsafe_allow_html=True)
