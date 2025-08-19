Colon Cancer Classifier Models Using Exome Sequencing Data

This repository contains a comprehensive Shiny for Python web application deployed on Posit Connect Cloud for the classification of colon cancer using annotated exome sequencing data. The interactive platform enables researchers and clinicians to upload genomic variant datasets, perform automated preprocessing, train multiple machine learning models simultaneously, and conduct comparative performance analysis through an intuitive web interface.
🌐 Live Application
Access the deployed application: https://0198c0d7-f1dd-f715-5964-e25554c67359.share.connect.posit.cloud/
📊 Overview
This Shiny for Python application provides a comprehensive machine learning pipeline for genomic variant analysis with the following capabilities:
Core Functionalities

Multi-file data upload with automated preprocessing (cleaning, label encoding, normalization)
Interactive data exploration with dynamic feature visualization and distribution analysis
Multi-model training using three complementary machine learning approaches:

🌳 Random Forest (scikit-learn) - Feature importance and ensemble learning
⚡ XGBoost - Gradient boosting for high-performance classification
🧠 Deep Neural Network (TensorFlow/Keras) - Complex pattern recognition


Real-time performance evaluation with accuracy metrics and confusion matrices
Comparative model analysis with side-by-side performance visualization
Comprehensive documentation and usage guidelines integrated into the interface

Interface Features

Multi-tab navigation for organized workflow management
Reactive programming for real-time updates and responsive interactions
Professional visualization using matplotlib and seaborn integration
Cloud-based deployment eliminating local computational requirements

🚀 Local Development Setup
Prerequisites

Python 3.8 or higher
pip package manager

Installation

Clone the repository:

bashgit clone https://github.com/CSK005/Colon-Cancer-AI-ML-model.git
cd Colon-Cancer-AI-ML-model

Create and activate virtual environment (recommended):

bashpython -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

Install required dependencies:

bashpip install -r requirements.txt
If requirements.txt is not available, install core dependencies manually:
bashpip install shiny pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
🖥️ Running the Application Locally
To launch the Shiny for Python application locally:
bashshiny run app.py
The application will be accessible at http://localhost:8000 (or the port specified in your configuration).
For development with auto-reload:
bashshiny run app.py --reload
📁 Input Data Format
The application accepts CSV files containing annotated exome sequencing data. Upload files should include genomic variant information with the following structure:
Required/Optional Columns:

Genomic coordinates: Chr (chromosome), Start, End
Functional annotations: Func.refGene (primary target variable), ExonicFunc.refGene
Pathogenicity scores: CADD, CADD_Phred, SIFT_pred, Polyphen2_HDIV_pred, Polyphen2_HVAR_pred
Prediction categories: MutationTaster_pred, MutationAssessor_pred, CLNSIG
Population frequencies: AF, AF_popmax

Automated Data Processing:

Missing value handling with intelligent imputation strategies
Categorical encoding using LabelEncoder for text-based annotations
Feature normalization with MinMaxScaler for numerical variables
Chromosome mapping converting chromosome identifiers to numeric values
Data validation and error handling for robust processing

Sample Data Format:
ChrStartEndFunc.refGeneCADDSIFT_predPolyphen2_HDIV_predAFchr11234512346exonic15.2TB0.001chr26789067891intronic8.7DP0.005
🤖 Machine Learning Models
1. Random Forest Classifier 🌳

Framework: scikit-learn
Configuration: 50 estimators, balanced class weights
Strengths: Feature importance analysis, handles missing values, reduces overfitting
Use Case: Exploratory analysis and feature selection

2. XGBoost Classifier ⚡

Framework: XGBoost Python API
Configuration: 50 estimators, mlogloss evaluation metric
Strengths: High accuracy, fast training, built-in regularization
Use Case: Production deployment and high-performance requirements

3. Deep Neural Network 🧠

Framework: TensorFlow/Keras
Architecture:

Input layer (dynamic feature count)
Hidden layer 1: 64 neurons (ReLU activation)
Hidden layer 2: 32 neurons (ReLU activation)
Output layer: Softmax (multi-class) or Sigmoid (binary)


Training: Adam optimizer, sparse categorical crossentropy loss
Use Case: Complex pattern recognition in large datasets

📈 Application Workflow
1. Data Upload and Preview

Upload training and testing CSV files (up to 200MB each)
Automatic data preview with tabular display
Dataset summary statistics and basic information

2. Exploratory Data Analysis

Interactive feature selection for visualization
Dynamic histogram and distribution plots
Real-time data quality assessment

3. Model Training and Evaluation

Simultaneous training of all three models
Real-time progress tracking and status updates
Automated hyperparameter configuration

4. Performance Analysis

Side-by-side accuracy comparison
Confusion matrices for each model
Visual performance metrics and statistical summaries

🎯 Key Features
Interactive Interface

Multi-tab navigation: Upload & Analyze, How to Use, Models Info, About
Responsive design: Compatible across devices and browsers
Real-time updates: Reactive programming with immediate feedback
Error handling: Comprehensive validation and user-friendly error messages

Scientific Rigor

Reproducible results: Fixed random seeds and documented methodologies
Transparent algorithms: Open-source implementation with detailed documentation
Performance metrics: Standard evaluation protocols with statistical significance
Data integrity: Robust preprocessing with quality control measures

👥 Research Team
Lead Developer & Data Scientist
Chandrashekar K
Department of Biotechnology, RV College of Engineering, Bengaluru, India
Principal Investigator & Research Supervisor
Dr. Vidya Niranjan
Department of Biotechnology, RV College of Engineering, Bengaluru, India
Research Collaborator
Anagha S Setlur
Department of Biotechnology, RV College of Engineering, Bengaluru, India
🔗 Links and Resources

🌐 Live Application: Posit Connect Cloud Deployment
📚 Documentation: Comprehensive usage guides available within the application
💻 Source Code: This GitHub repository
📧 Contact: For research collaborations and technical support

📄 Citation
If this application contributes to your research, please cite:
bibtex@software{colon_cancer_classifier_2024,
  author = {Chandrashekar, K. and Niranjan, Vidya and Setlur, Anagha S.},
  title = {Colon Cancer Classifier Models Using Exome Sequencing Data},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/CSK005/Colon-Cancer-AI-ML-model},
  note = {Shiny for Python web application deployed on Posit Connect Cloud}
}
