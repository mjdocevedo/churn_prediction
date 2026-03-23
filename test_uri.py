import mlflow, os
mlflow.set_tracking_uri("http://localhost:5000")
print("ENV tracking URI:", os.environ.get("MLFLOW_TRACKING_URI"))
print("Current tracking URI:", mlflow.get_tracking_uri())
print("Registry URI:", mlflow.get_registry_uri())
