import mlflow
import os
from mlflow.tracking import MlflowClient

# --- Configuration ---
# In a pipeline, we pass this via environment variable.
# For manual testing, you can paste a run_id here.
RUN_ID = os.getenv("MLFLOW_RUN_ID") 
MODEL_NAME = "ChurnModel"

def register():
    if not RUN_ID:
        print("Please set the MLFLOW_RUN_ID environment variable or update the script.")
        return

    print(f"Registering model from Run ID: {RUN_ID} as '{MODEL_NAME}'...")
    
    # 1. Register Model
    model_uri = f"runs:/{RUN_ID}/model"
    model_details = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    
    print(f"Model registered. Version: {model_details.version}")
    
    # 2. Transition to Staging
    client = MlflowClient()
    print(f"Transitioning version {model_details.version} to Staging...")
    
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_details.version,
        stage="Staging"
    )
    print("Transition complete.")

if __name__ == "__main__":
    register()
