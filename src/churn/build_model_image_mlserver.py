import mlflow

# --- Configuration ---
# We build the image for the model currently in 'Production'
MODEL_NAME = "ChurnModel"
STAGE = "Production"
DOCKER_IMAGE_NAME = "churn-model-production-mlserver"

def build_model_image_mlserver():
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    print(f"=== Building Docker Image (MLServer) for {model_uri} ===")
    print(f"Target Image Name: {DOCKER_IMAGE_NAME}")
    print(f"Inference Server: MLServer (Kubernetes-optimized)\n")
    
    try:
        # MLflow built-in function to generate a Docker image containing the model
        # Using MLServer for production-grade, scalable inference
        # Reference: https://mlflow.org/docs/latest/python_api/mlflow.models.html#mlflow.models.build_docker
        mlflow.models.build_docker(
            model_uri=model_uri,
            name=DOCKER_IMAGE_NAME,
            enable_mlserver=True,  # Use MLServer for production scalability
        )
        
        print(f"\n✓ SUCCESS: Docker image '{DOCKER_IMAGE_NAME}' built successfully.")
        print(f"Inference Server: MLServer (production-grade)")
        print(f"\nRun it with:")
        print(f"  docker run -p 8080:8080 {DOCKER_IMAGE_NAME}")
        print(f"\nTest the inference endpoint:")
        print(f"  curl -X POST -H 'Content-Type: application/json' \\")
        print(f"    --data '{{\"inputs\": [[...]]}}' \\")
        print(f"    http://localhost:8080/invocations")
        
    except Exception as e:
        print(f"\n✗ ERROR: Failed to build image.")
        print(f"Details: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure Docker daemon is running")
        print(f"2. Verify model '{MODEL_NAME}' exists in stage '{STAGE}'")
        print("3. Check that the model has actual artifacts (not just metadata)")
        print("4. Ensure MLServer dependencies are installed")

if __name__ == "__main__":
    build_model_image_mlserver()