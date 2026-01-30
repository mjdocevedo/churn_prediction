import mlflow
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from loader import get_train_test_split_data

# --- Configuration ---
# To evaluate, you need a Model URI. 
# After running train.py, replace this with the URI from your latest run or Model Registry.
# Example: "runs:/<RUN_ID>/model" or "models:/ChurnModel/1"
MODEL_URI = "models:/ChurnModel/Production" 
DATA_PATH = "data/telco_churn.csv"

def evaluate():
    print(f"Loading test data from {DATA_PATH}...")
    _, X_test, _, y_test = get_train_test_split_data(DATA_PATH)
    
    print(f"Loading model from {MODEL_URI}...")
    try:
        model = mlflow.sklearn.load_model(MODEL_URI)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please check the MODEL_URI in evaluate.py")
        return

    print("Running predictions...")
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy: {acc:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

if __name__ == "__main__":
    evaluate()
