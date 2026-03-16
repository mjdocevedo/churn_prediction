import mlflow
import os
from mlflow.tracking import MlflowClient

# --- Configuration ---
# In a pipeline, we pass this via environment variable.
# For manual testing, we try to detect the latest run.
MODEL_NAME = "ChurnModel"
EXPERIMENT_NAME = "Churn_Prediction_Basic"

def get_latest_run_id():
    """Detect the last successful run in the experiment."""
    try:
        last_run = mlflow.search_runs(
            experiment_names=[EXPERIMENT_NAME], 
            filter_string="tags.mlflow.runName = 'Model_Training'",
            order_by=["start_time DESC"], 
            max_results=1
        )
        if not last_run.empty:
            return last_run.iloc[0].run_id
    except Exception:
        pass
    return None

def register(run_id=None):
    # Evaluate at runtime to pick up values set by pipeline.py
    if not run_id:
        #run_id = os.getenv("MLFLOW_RUN_ID") 
        pass
    
    if not run_id:
        print("MLFLOW_RUN_ID not set. Attempting to auto-detect latest run...")
        run_id = get_latest_run_id()
        
    if not run_id:
        print("Error: Could not find any runs to register. Please run train.py first.")
        return

    print(f"Registering model from Run ID: {run_id} as '{MODEL_NAME}'...")
    
    # 1. Register Model
    # Inside Docker (mlflow run), the volume is mounted at /mlflow/tmp/mlruns
    # But the metadata stores the host path. We need to help MLflow find it.
    if os.path.exists("/mlflow/tmp/mlruns"):
        model_uri = f"file:///mlflow/tmp/mlruns/{mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id}/{run_id}/artifacts/model"
    else:
        model_uri = f"runs:/{run_id}/model"
    
    client = MlflowClient()
    
    # Check if model already exists, if not create it
    try:
        client.get_registered_model(MODEL_NAME)
    except Exception:
        print(f"Creating registered model '{MODEL_NAME}'...")
        client.create_registered_model(MODEL_NAME)

    model_details = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run_id
    )
    
    print(f"Model registered. Version: {model_details.version}")
    # 2. Transition to Staging
    print(f"Transitioning version {model_details.version} to Staging...")
    
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_details.version,
        stage="Staging"
    )
    print("Transition complete.")

if __name__ == "__main__":
    register()