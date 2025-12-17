import pandas as pd
from sklearn.model_selection import train_test_split
import os

# --- 1. Define Folder and File Constants ---
# ASSUMPTION: The folder is named 'data' (lowercase).
# If your folder is named 'DATA' (uppercase), change this to 'DATA'.
DATA_FOLDER_NAME = 'Data' 
FILE_NAME = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

# Full path to the source CSV
FULL_SOURCE_PATH = os.path.join(DATA_FOLDER_NAME, FILE_NAME) 

# Full paths for the output files
X_TRAIN_FILE = os.path.join(DATA_FOLDER_NAME, 'X_train.csv')
X_TEST_FILE = os.path.join(DATA_FOLDER_NAME, 'X_test.csv')
Y_TRAIN_FILE = os.path.join(DATA_FOLDER_NAME, 'y_train.csv')
Y_TEST_FILE = os.path.join(DATA_FOLDER_NAME, 'y_test.csv')


# --- 2. Ensure the DATA folder exists ---
if not os.path.exists(DATA_FOLDER_NAME):
    print(f"Creating directory: {DATA_FOLDER_NAME}/")
    os.makedirs(DATA_FOLDER_NAME)

def ingest_data(file_path):
    """Loads, cleans, and splits the Telco Churn data into train/test sets."""
    
    # 1. Load the dataset
    try:
        df = pd.read_csv(file_path)
        # Using os.path.abspath helps confirm the path being used
        print(f"Attempting to load data from: {os.path.abspath(file_path)}") 
    except FileNotFoundError:
        # Note the file_path printed in the error is the full path the system tried to use
        print(f"ERROR: Source file '{file_path}' not found.")
        print(f"Please confirm the file and the folder name ('{DATA_FOLDER_NAME}') are spelled correctly and cased properly.")
        return 


    # 2. Initial Clean: Drop non-predictive columns
    df = df.drop(columns=['customerID'], errors='ignore')

    # 3. CRITICAL CLEANING: Fix TotalCharges string issue
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # 4. Handle Missing Values: Drop the 11 rows with NaN TotalCharges
    df.dropna(inplace=True)

    # 5. Separate features (X) and target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0}) 

    # 6. Correct Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42, 
        stratify=y       
    )
    
    # 7. Save the split data to CSV in the DATA folder
    X_train.to_csv(X_TRAIN_FILE, index=False)
    X_test.to_csv(X_TEST_FILE, index=False)
    y_train.to_csv(Y_TRAIN_FILE, index=False, header=['Churn'])
    y_test.to_csv(Y_TEST_FILE, index=False, header=['Churn'])
    
    print("-" * 50)
    print(f"Data ingestion and split complete.")
    print(f"Output files saved to the **{DATA_FOLDER_NAME}/** folder.")
    print(f"Train/Test shapes: X_train={X_train.shape}, X_test={X_test.shape}")
    print("-" * 50)

if __name__ == "__main__":
    ingest_data(FULL_SOURCE_PATH)