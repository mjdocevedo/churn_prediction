import mlflow
from mlflow.tracking import MlflowClient

# --- Configuration ---
# We build the image for the model currently in 'Production'
MODEL_NAME = "ChurnModel"
STAGE = "Production"
DOCKER_IMAGE_NAME = "churn-model-production"


def _get_stage_model_version(client: MlflowClient):
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    stage_versions = [v for v in versions if v.current_stage == STAGE]
    if not stage_versions:
        return None
    return max(stage_versions, key=lambda v: int(v.version))


def build_model_image():
    client = MlflowClient()
    stage_version = _get_stage_model_version(client)

    if not stage_version:
        print(f"No model found in stage '{STAGE}' for '{MODEL_NAME}'.")
        print("Run: make workflow")
        return

    # More robust with local file store than models:/name/stage in this setup
    model_uri = f"runs:/{stage_version.run_id}/model"

    print(f"=== Building Docker Image for {MODEL_NAME} v{stage_version.version} ({STAGE}) ===")
    print(f"Using URI: {model_uri}")
    print(f"Target Image Name: {DOCKER_IMAGE_NAME}")

    try:
        mlflow.models.build_docker(
            model_uri=model_uri,
            name=DOCKER_IMAGE_NAME,
            enable_mlserver=False,
        )

        print(f"\n✓ SUCCESS: Docker image '{DOCKER_IMAGE_NAME}' built successfully.")
        print(f"Run it with: docker run -p 5000:8080 {DOCKER_IMAGE_NAME}")

    except Exception as e:
        print(f"\n✗ ERROR: Failed to build image.")
        print(f"Details: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Docker daemon is running")
        print(f"2. Verify model '{MODEL_NAME}' exists in stage '{STAGE}'")
        print("3. Check that the model has actual artifacts (not just metadata)")


if __name__ == "__main__":
    build_model_image()