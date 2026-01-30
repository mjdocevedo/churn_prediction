import mlflow
import subprocess
import os

def run_pipeline():
    print("=== Starting MLOps Pipeline ===")
    
    # 1. Train
    print("\n[Step 1] Training Model...")
    subprocess.run(["uv", "run", "src/train.py"], check=True)
    
    # Retrieve the last run ID
    last_run = mlflow.search_runs(experiment_names=["Churn_Prediction_Basic"], order_by=["start_time DESC"], max_results=1).iloc[0]
    run_id = last_run.run_id
    print(f"Training Complete. Run ID: {run_id}")
    
    # 2. Evaluate
    print("\n[Step 2] Evaluating Model...")
    model_uri = f"runs:/{run_id}/model"
    
    # Pass configuration via environment variables
    env = os.environ.copy()
    env["MLFLOW_MODEL_URI_OVERRIDE"] = model_uri
    
    subprocess.run(["uv", "run", "src/evaluate.py"], check=True, env=env)
    
    # 3. Register
    print("\n[Step 3] Registering Model...")
    env["MLFLOW_RUN_ID"] = run_id
    subprocess.run(["uv", "run", "src/register.py"], check=True, env=env)
    
    # 4. Serve
    print("\n[Step 4] Serving Model (Instruction)...")
    print(f"Model from run {run_id} is now registered in 'Staging'.")
    print("To serve locally, run:")
    print("mlflow models serve -m models:/ChurnModel/Staging -p 5000")

if __name__ == "__main__":
    run_pipeline()
