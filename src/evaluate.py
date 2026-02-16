import mlflow
import os
import pandas as pd
from loader import get_train_test_split_data
import logging
import warnings

# --- Setup Logging ---
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="mlflow")
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

# --- Configuration ---
DATA_PATH = "data/telco_churn.csv"
EXPERIMENT_NAME = "Churn_Prediction_Basic"

def get_latest_run_id():
    try:
        # Insert your code here
        # Search for the latest run in the experiment
        last_run = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME], 
            order_by=["start_time DESC"], 
            max_results=1
        )
        if not last_run.empty:
            return last_run.iloc[0].run_id
    except Exception:
        pass
    return None

def evaluate():
    # Determine Model URI
    # Priority: Env Var > Latest Run > Placeholder
    env_uri = os.getenv("MLFLOW_MODEL_URI_OVERRIDE")
    if env_uri:
        model_uri = env_uri
    else:
        latest_run_id = get_latest_run_id()
        if latest_run_id:
            print(f"Auto-detected latest run: {latest_run_id}")
            model_uri = f"runs:/{latest_run_id}/model"
        else:
            model_uri = "runs:/<REPLACE_WITH_YOUR_RUN_ID>/model"

    print(f"Loading test data from {DATA_PATH}...")
    _, X_test, _, y_test = get_train_test_split_data(DATA_PATH)
    
    # Combine for mlflow.evaluate
    eval_data = X_test.copy()
    eval_data["Churn"] = y_test
    
    print(f"Evaluating model: {model_uri}")

    # Insert your code here
    # Ensure we log to the same experiment as training
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    with mlflow.start_run(run_name="Model_Evaluation"):
        # Insert your code here
        # Use mlflow.models.evaluate
        result = mlflow.evaluate(
            model=model_uri,
            data=eval_data,
            targets="Churn",
            model_type="classifier",
            evaluators=["default"],
        )
        
        print("\nEvaluation metrics logged to MLflow:")
        # Print a clean subset of metrics
        metrics_to_show = ["accuracy_score", "f1_score", "roc_auc"]
        clean_metrics = {k: v for k, v in result.metrics.items() if k in metrics_to_show}
        print(clean_metrics)

if __name__ == "__main__":
    evaluate()
