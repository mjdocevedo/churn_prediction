import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def load_data(path="data/telco_churn.csv"):
    """
    Load and preprocess the Telco Churn dataset.
    Performs cleanup, categorical encoding, and type conversion.
    """
    # 1. Load Data
    df = pd.read_csv(path)
    
    # 2. Cleanup: Helper function to clean numeric columns
    # 'TotalCharges' contains empty strings ' ' for new customers (tenure=0).
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    # 3. Drop irrelevant columns
    # customerID is unique and not predictive
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        
    # 4. Target Encoding
    # Churn: Yes -> 1, No -> 0
    if 'Churn' in df.columns:
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
        
    # 5. Categorical Encoding
    # Identify categorical columns (object type)
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # For linear models and tree-based models, One-Hot Encoding is a safe bet for low-cardinality nominal vars.
    # We use drop_first=True to avoid multicollinearity for linear models (dummy variable trap).
    df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    
    return df

def get_train_test_split_data(path="data/telco_churn.csv", test_size=0.2, random_state=42):
    """
    Loads data and returns X_train, X_test, y_train, y_test.
    """
    df = load_data(path)
    
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
