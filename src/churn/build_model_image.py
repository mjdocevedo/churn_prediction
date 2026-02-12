import mlflow
import os

# --- Configuration ---
# We build the image for the model currently in 'Staging'
MODEL_NAME = "ChurnModel"
STAGE = "Staging"
DOCKER_IMAGE_NAME = "churn-model:production"

def build_model_image():
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    print(f"=== Building Docker Image for {model_uri} ===")
    print(f"Target Image Name: {DOCKER_IMAGE_NAME}")
    
    try:
        # MLflow built-in function to generate a Docker image containing the model
        mlflow.models.build_docker(
            model_uri=model_uri,
            name=DOCKER_IMAGE_NAME,
            enable_mlserver=True, # Improved serving performance
        )
        print(f"\nSUCCESS: Image '{DOCKER_IMAGE_NAME}' built.")
        print(f"Run it with: docker run -p 5000:8080 {DOCKER_IMAGE_NAME}")
        
    except Exception as e:
        print(f"\nERROR: Failed to build image. Ensure Docker is running.\n{e}")

if __name__ == "__main__":
    build_model_image()
