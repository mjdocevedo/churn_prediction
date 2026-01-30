import mlflow
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from loader import get_train_test_split_data

# --- Configuration ---
DATA_PATH = "data/telco_churn.csv"
N_ESTIMATORS = 100
MAX_DEPTH = 10
EXPERIMENT_NAME = "Churn_Prediction_Basic"

def train():
    # 1. Load Data
    print("Loading data...")
    X_train, X_test, y_train, y_test = get_train_test_split_data(DATA_PATH)

    # 2. Setup MLflow
    # MLflow 3.x style: set experiment explicitly
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Enable Autologging
    # This captures params, metrics, model artifacts, and system metrics automatically
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    with mlflow.start_run():
        print("Starting training run...")
        
        # 3. Model Training
        rf = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            max_depth=MAX_DEPTH,
            random_state=42
        )
        rf.fit(X_train, y_train)
        
        # 4. Custom Metrics/Artifacts (Extension to autologging)
        # Even with autolog, we might want custom plots
        print("Logging custom artifacts...")
        
        # Confusion Matrix Plot
        y_pred = rf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Save plot to MLflow
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        print("Run complete!")

if __name__ == "__main__":
    train()
