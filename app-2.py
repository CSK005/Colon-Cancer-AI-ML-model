import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from tensorflow import keras
from tensorflow.keras import layers

# Load data
df = pd.read_csv("your_data.csv")  # Replace with actual file

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Apply Label Encoding to categorical features
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Ensure all data is numeric
df = df.apply(pd.to_numeric, errors='coerce')
df.fillna(0, inplace=True)  # Fill NaN values with 0

# Split data
X = df.drop(columns=['Target'])  # Replace 'Target' with your actual target column
y = df['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train RandomForest
try:
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    print("RandomForest trained successfully")
except Exception as e:
    print(f"Error training RandomForest: {e}")

# Train XGBoost
try:
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    params = {'objective': 'binary:logistic', 'enable_categorical': True}
    xgb_model = xgb.train(params, dtrain, num_boost_round=100)
    print("XGBoost trained successfully")
except Exception as e:
    print(f"Error training XGBoost: {e}")

# Train Deep Neural Network (DNN)
try:
    model = keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))
    print("DNN trained successfully")
except Exception as e:
    print(f"Error training DNN model: {e}")
