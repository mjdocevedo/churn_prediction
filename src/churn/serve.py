import os
import subprocess
import time

# --- Configuration ---
# In a real setup, we might pull this from the Registry (Staging)
MODEL_URI = os.getenv("MLFLOW_MODEL_URI_OVERRIDE", "models:/ChurnModel/Staging")
PORT = 5000

def serve():
    print(f"Starting Model Server for: {MODEL_URI} on port {PORT}...")
    print("Press Ctrl+C to stop.")
    
    # We use subprocess to call the CLI. 
    # This wrapper allows us to set defaults and handle environment setup easily.
    cmd = [
        "mlflow", "models", "serve",
        "--model-uri", MODEL_URI,
        "--host", "0.0.0.0",
        "--port", str(PORT),
        "--no-conda" 
    ]
    
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("Stopping server...")

if __name__ == "__main__":
    serve()
