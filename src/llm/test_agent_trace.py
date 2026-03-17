import mlflow
import os
from agent import create_retention_agent
from dotenv import load_dotenv

load_dotenv(override=True)

def verify_traces():
    """
    Manually triggers a run to verify that traces are correctly captured 
    with the required spans (churn_risk, retention_rules, generate).
    """
    # 1. Setup MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("llmops_retention_agent")

    # 2. Initialize Agent
    agent = create_retention_agent(prompt_version="v0.1")

    # 3. Trigger Trace
    # Scenario: A customer with a specific ID asking for a discount
    test_query = "Je suis client depuis 3 ans (ID: 7590-VHVEG), est-ce que j'ai droit à une remise ?"
    
    print(f"Starting trace verification for query: '{test_query}'...")
    
    with mlflow.start_run(run_name="trace_verification_v0.1"):
        mlflow.set_tags({
            "prompt_version": "v0.1",
            "churn_model_version": "Production",
            "rules_version": "v1.0",
            "env": "dev",
            "verification_type": "manual_trace"
        })
        
        response = agent.invoke({"input": test_query})
        
    print("\n--- AGENT OUTPUT ---")
    print(response["output"])
    print("\n--- TRACE VERIFIED ---")
    print("Check MLflow UI (http://localhost:5000) to confirm spans: get_churn_risk, retrieve_retention_rules, generate.")

if __name__ == "__main__":
    verify_traces()
