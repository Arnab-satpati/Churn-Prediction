import pandas as pd
import joblib 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
import os

# --- 1. File Constants and Data Loading ---
X_TRAIN_FILE = 'Data/X_train.csv'
X_TEST_FILE = 'Data/X_test.csv'
Y_TRAIN_FILE = 'Data/y_train.csv'
Y_TEST_FILE = 'Data/y_test.csv'
RF_MODEL_FILE = 'rf_pipeline.pkl'

print("--- 03_train.py: Starting Model Training ---")

try:
    X_train = pd.read_csv(X_TRAIN_FILE)
    X_test = pd.read_csv(X_TEST_FILE)
    # The target column is loaded as a Series
    y_train = pd.read_csv(Y_TRAIN_FILE)['Churn']
    y_test = pd.read_csv(Y_TEST_FILE)['Churn']
except FileNotFoundError as e:
    print(f"ERROR: Could not find split data files. Ensure 01_ingest.py ran successfully. {e}")
    exit()

# --- 2. Define Preprocessor Architecture (ML Engineering Robustness) ---
# We redefine the preprocessor structure here for a fully self-contained script.

numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']
categorical_features = [
    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod'
]

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop'
)

# --- 3. Define the Production Model (Random Forest) ---

# CRITICAL: Use class_weight='balanced' to handle the Churn imbalance 
# (This assigns higher weight to the minority Churn class).
rf_model = RandomForestClassifier(
    n_estimators=300,        
    max_depth=10,            
    random_state=42,
    class_weight='balanced' 
)

# --- 4. Assemble the Full Deployable Pipeline ---
# This single object links preprocessing and classification.
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])

# --- 5. Train and Evaluate ---
print("\n--- Training Random Forest Pipeline ---")

# .fit() on the pipeline automatically runs preprocessor.fit_transform on X_train 
# and then classifier.fit() on the transformed data.
rf_pipeline.fit(X_train, y_train)

# Predict on the test set (runs preprocessor.transform + classifier.predict)
y_pred_rf = rf_pipeline.predict(X_test)
y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1] 

# --- 6. Report Production Metrics ---
print("\n[Hirable Metric Report: Random Forest Model]")
report_rf = classification_report(y_test, y_pred_rf, target_names=['No Churn (0)', 'Churn (1)'])
print(report_rf)
auc_score_rf = roc_auc_score(y_test, y_proba_rf)
print(f"ROC-AUC Score: {auc_score_rf:.4f}")

# --- 7. Save the Complete Pipeline ---
# This single file is the only thing needed for deployment (it's the whole model).
joblib.dump(rf_pipeline, RF_MODEL_FILE)

print(f"\nSUCCESS: Complete Random Forest Pipeline saved to '{RF_MODEL_FILE}'.")