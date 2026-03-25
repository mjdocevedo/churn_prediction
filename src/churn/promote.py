"""
promote.py — Metric-based Model Promotion

Promotes a model version from Staging to Production only if it meets
a minimum quality threshold (e.g., F1 score).

This demonstrates a key MLOps concept: automated governance.
In production, this logic often lives in a CI/CD pipeline (GitHub Actions,
GitLab CI, Jenkins) rather than being run manually.
"""
import mlflow
from mlflow.tracking import MlflowClient
import subprocess
import logging
import warnings
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv(override=True)

# --- Setup Logging ---
logging.getLogger("mlflow").setLevel(logging.ERROR)
logging.getLogger("alembic").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning, module="mlflow")

# --- Configuration ---
MODEL_NAME = "ChurnModel"
F1_THRESHOLD = 0.50  # Minimum F1 score to promote to Production
EXPERIMENT_NAME = "Churn_Prediction_Basic"


def get_staging_model_version(client):
    """Find the latest model version in Staging."""
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    staging_versions = [v for v in versions if v.current_stage == "Staging"]
    if not staging_versions:
        return None
    # Return the most recent staging version
    return max(staging_versions, key=lambda v: int(v.version))


def get_evaluation_metrics(run_id):
    """Retrieve F1 score from the evaluation run associated with a model."""
    # Search for evaluation runs that evaluated this model
    # Insert your code here
    eval_runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"tags.mlflow.runName = 'Model_Evaluation'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    if eval_runs.empty:
        return None
    
    f1 = eval_runs.iloc[0].get("metrics.f1_score")
    accuracy = eval_runs.iloc[0].get("metrics.accuracy_score")
    return {"f1_score": f1, "accuracy_score": accuracy}


def get_git_sha():
    """Get current git commit SHA for traceability."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def promote():
    client = MlflowClient()

    # 1. Find the model in Staging (We still check Staging to find the candidate)
    staging_version = get_staging_model_version(client)
    if not staging_version:
        print("No model version found in 'Staging'.")
        return

    # 2. Check evaluation metrics
    metrics = get_evaluation_metrics(staging_version.run_id)
    if metrics is None:
        print("No evaluation metrics found.")
        return

    f1 = metrics.get("f1_score", 0)
    
    # 3. Decision
    if f1 >= F1_THRESHOLD:
        print(f"\nPROMOTING version {staging_version.version} to Champion!")

        # Set Tags for traceability
        client.set_model_version_tag(MODEL_NAME, staging_version.version, "git_sha", get_git_sha())

        # NEW: Assign the 'champion' alias
        # This is what the model-server will look for
        client.set_registered_model_alias(
            name=MODEL_NAME, 
            alias="champion", 
            version=staging_version.version
        )
        
        # Optional: Keep the stage transition for UI visibility
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=staging_version.version,
            stage="Production",
            archive_existing_versions=True
        )
        
        print(f"Model version {staging_version.version} is now tagged as @champion")
    else:
        print(f"F1 score {f1:.4f} too low. No promotion.")

if __name__ == "__main__":
    promote()