import mlflow
import os
import pandas as pd
from loader import get_train_test_split_data

# --- Configuration ---
MODEL_URI = os.getenv("MLFLOW_MODEL_URI_OVERRIDE", "models:/ChurnModel/Production") 
DATA_PATH = "data/telco_churn.csv"

def evaluate():
    print(f"Loading test data from {DATA_PATH}...")
    # Load separate X and y
    _, X_test, _, y_test = get_train_test_split_data(DATA_PATH)
    
    # Combine for mlflow.evaluate which expects a single dataset with specific targets
    eval_data = X_test.copy()
    eval_data["Churn"] = y_test
    
    print(f"Evaluating model: {MODEL_URI}")
    
    # Start a new run for evaluation (or log to existing one if we had the ID)
    # Since this is a distinct 'evaluate' step in the pipeline, a new run makes sense for observability.
    with mlflow.start_run(run_name="Evaluation"):
        # mlflow.evaluate is the "Model-First" approach to evaluation
        result = mlflow.evaluate(
            model=MODEL_URI,
            data=eval_data,
            targets="Churn",
            model_type="classifier",
            evaluators=["default"],
        )
        
        print("\nEvaluation Result:")
        print(f"Metrics: {result.metrics}")
        print(f"Artifacts: {result.artifacts}")

if __name__ == "__main__":
    evaluate()
