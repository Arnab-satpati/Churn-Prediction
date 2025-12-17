import pandas as pd
import joblib # Better than pickle for sklearn objects
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# --- 1. Load the Split Data ---
# Assuming 01_ingest.py has run successfully and created these files.
try:
    X_train = pd.read_csv('Data/X_train.csv')
    y_train = pd.read_csv('Data/y_train.csv')['Churn']
    # FIX: Load X_test here
    X_test = pd.read_csv('Data/X_test.csv') 
    # y_test is not needed for preprocessing but is good practice to load if available
    # y_test = pd.read_csv('data/y_test.csv')['Churn'] 

except FileNotFoundError:
    print("Error: Split data files (X_train.csv, X_test.csv, etc.) not found.")
    print("Please run 01_ingest.py successfully first, and ensure the data folder exists.")
    raise

# --- 2. Define Feature Types ---
# Check the notebook/dataset columns to get these lists right.
numerical_features = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Exclude binary columns handled by OHE later.
categorical_features = [
    'gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
    'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen'
]

# --- 3. Define Preprocessing Steps (Preprocessors) ---

# Preprocessor for Numerical Features:
# 1. Impute missing values with the median of the training set.
# 2. Scale features using Standardization (mean=0, std=1).
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessor for Categorical Features:
# 1. Impute missing values with the most frequent category.
# 2. Apply One-Hot Encoding.
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
])

# --- 4. Combine Preprocessors into ColumnTransformer ---
# This applies the right transformer to the right column set.
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' 
)

# --- 5. Create the Full Preprocessing Pipeline (Fit and Transform) ---
print("Fitting preprocessing pipeline on X_train...")

# We fit the transformer and transform the training data.
# The 'fit' step calculates the median/most_frequent values, and mean/std.
X_train_transformed = preprocessor.fit_transform(X_train)

# Transform the test data (CRITICAL: DO NOT use fit_transform here)
# The 'transform' step applies the values learned from X_train to X_test.
X_test_transformed = preprocessor.transform(X_test)

# --- 6. Save the Fitted Preprocessor ---
# This fitted object contains all the means, medians, and category mappings needed for deployment.
joblib.dump(preprocessor, 'models/preprocessor.pkl')

print("Preprocessing complete.")
print(f"X_train transformed shape: {X_train_transformed.shape}")
print(f"X_test transformed shape: {X_test_transformed.shape}")
print("Fitted preprocessor pipeline saved to 'models/preprocessor.pkl'.")