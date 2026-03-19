import mlflow
from sklearn.ensemble import RandomForestClassifier
from churn.loader import get_train_test_split_data
from mlflow.models import infer_signature
import os
import logging
import warnings
import tempfile
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(override=True)

MODEL_NAME = "ChurnModel"

# --- Configuration ---
DATA_PATH = "data/telco_churn.csv"
N_ESTIMATORS = 100
MAX_DEPTH = 10
EXPERIMENT_NAME = "Churn_Prediction_Basic"

# --- Setup Logging ---
# Silence verbose MLflow and Alembic logs
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)
# Ignore specific warnings
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

def train():
    # 1. Load Data
    print("Loading data...")
    X_train, X_test, y_train, y_test = get_train_test_split_data(DATA_PATH)

    # 2. Setup MLflow
    # MLflow 3.x style: set experiment explicitly

    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Enable Autologging
    # This captures params, metrics, model artifacts, and system metrics automatically
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True, silent=True)

    with mlflow.start_run(run_name="Model_Training"):
        print("Starting training run...")
        
        # 3. Model Training
        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        # 4. Log model artifact at artifacts/model for downstream packaging
        signature = infer_signature(X_train, rf.predict(X_train))
        mlflow.sklearn.log_model(
            rf, 
            "model", 
            signature=signature,
            registered_model_name=MODEL_NAME
        )
        
        print(f"Run complete! Model logged to 'mlruns'")
        return mlflow.active_run().info.run_id

if __name__ == "__main__":
    train()