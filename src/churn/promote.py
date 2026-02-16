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
    # Insert your code here
    client = MlflowClient()

    # 1. Find the model in Staging
    staging_version = get_staging_model_version(client)
    if not staging_version:
        print("No model version found in 'Staging'. Run the pipeline first.")
        return

    print(f"Found model '{MODEL_NAME}' version {staging_version.version} in Staging.")
    print(f"  Source Run: {staging_version.run_id}")

    # 2. Check evaluation metrics
    metrics = get_evaluation_metrics(staging_version.run_id)
    if metrics is None:
        print("No evaluation metrics found. Run evaluate.py first.")
        return

    f1 = metrics.get("f1_score", 0)
    accuracy = metrics.get("accuracy_score", 0)
    print(f"\n Evaluation Metrics:")
    print(f"  F1 Score:  {f1:.4f}  (threshold: {F1_THRESHOLD})")
    print(f"  Accuracy:  {accuracy:.4f}")

    # 3. Decision: Promote or Reject
    if f1 >= F1_THRESHOLD:
        print(f"\n F1 score ({f1:.4f}) >= threshold ({F1_THRESHOLD}). PROMOTING to Production!")

        # Add traceability tags
        git_sha = get_git_sha()
        client.set_model_version_tag(MODEL_NAME, staging_version.version, "git_sha", git_sha)
        client.set_model_version_tag(MODEL_NAME, staging_version.version, "promoted_by", "promote.py")
        client.set_model_version_tag(MODEL_NAME, staging_version.version, "f1_at_promotion", str(round(f1, 4)))

        # Insert your code here
        # Transition to Production
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=staging_version.version,
            stage="Production",
            archive_existing_versions=True  # Archive previous production versions
        )
        print(f"Model version {staging_version.version} is now in Production!")
        print(f"   Tagged with git_sha={git_sha}")
    else:
        print(f"\n F1 score ({f1:.4f}) < threshold ({F1_THRESHOLD}). NOT promoting.")
        print("   Improve your model or lower the threshold.")


if __name__ == "__main__":
    promote()
