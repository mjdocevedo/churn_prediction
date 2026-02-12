import mlflow
from sklearn.ensemble import RandomForestClassifier
from churn.loader import get_train_test_split_data
import os
import logging
import warnings

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
        
        print(f"Run complete! Artifacts saved to 'mlruns'")
        return mlflow.active_run().info.run_id

if __name__ == "__main__":
    train()
