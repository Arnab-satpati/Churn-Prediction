import pandas as pd
import joblib 
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

# --- 1. File Constants and Data Loading ---
X_TRAIN_FILE = 'Data/X_train.csv'
X_TEST_FILE = 'Data/X_test.csv'
Y_TRAIN_FILE = 'Data/y_train.csv'
Y_TEST_FILE = 'Data/y_test.csv'
PREPROCESSOR_FILE = 'models/preprocessor.pkl'  # The existing file
RF_PIPELINE_FILE = 'models/rf_pipeline.pkl'    # The final combined model

print("--- 03_train.py: Starting Model Training ---")

try:
    X_train = pd.read_csv(X_TRAIN_FILE)
    X_test = pd.read_csv(X_TEST_FILE)
    y_train = pd.read_csv(Y_TRAIN_FILE)['Churn']
    y_test = pd.read_csv(Y_TEST_FILE)['Churn']
    
    # LOAD the existing preprocessor
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    print(f"SUCCESS: Loaded preprocessor from {PREPROCESSOR_FILE}")
except FileNotFoundError as e:
    print(f"ERROR: Missing files. Check Data/ folder or models/preprocessor.pkl. {e}")
    exit()

# --- 2. Define the Production Model (Random Forest) ---
rf_model = RandomForestClassifier(
    n_estimators=300,        
    max_depth=10,            
    random_state=42,
    class_weight='balanced' 
)

# --- 3. Assemble the Full Deployable Pipeline ---
# We link the LOADED preprocessor directly to the new classifier
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', rf_model)
])

# --- 4. Train and Evaluate ---
print("\n--- Training Random Forest Pipeline ---")

# Note: Since the preprocessor was likely already 'fit' in your 
# previous script, the Pipeline will re-fit it here to be safe, 
# or you could use it to just transform if you were strictly predicting.
rf_pipeline.fit(X_train, y_train)

# Predict
y_pred_rf = rf_pipeline.predict(X_test)
y_proba_rf = rf_pipeline.predict_proba(X_test)[:, 1] 

# --- 5. Report Production Metrics ---
print("\n[Hirable Metric Report: Random Forest Model]")
print(classification_report(y_test, y_pred_rf, target_names=['No Churn (0)', 'Churn (1)']))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba_rf):.4f}")

# --- 6. Save the Complete Pipeline ---
joblib.dump(rf_pipeline, RF_PIPELINE_FILE)
print(f"\nSUCCESS: Full Pipeline (Preprocessor + RF) saved to '{RF_PIPELINE_FILE}'.")