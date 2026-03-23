import mlflow
import os
import sys
import pandas as pd
import json
from pathlib import Path
from mlflow.genai.scorers import scorer
from dotenv import load_dotenv

load_dotenv(override=True)

# Ensure the project root is on sys.path when running this module directly.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.llm.agent import create_retention_agent

# 1. Custom Scorer: JSON Format Compliance
def json_format_ok(output: str) -> float:
    """Verifies that the output is valid JSON and contains required keys."""
    required_keys = ["customer_id", "risk", "offer", "justification", "sources"]
    try:
        parsed = json.loads(output)
        if all(k in parsed for k in required_keys):
            return 1.0
        return 0.5  # Valid JSON but missing keys
    except:
        return 0.0

# 2. Custom Scorer: Discount Policy Compliance (Manager's Absolute Rule)
def discount_policy_compliance(output: str) -> float:
    """
    Zero tolerance for policy violations: the agent must never invent a discount.
    If 'offer' is present, there must be a 'justification' and 'sources'.
    """
    try:
        parsed = json.loads(output)
        offer = parsed.get("offer")
        sources = parsed.get("sources", [])
        
        # If no offer, it's compliant (assuming the agent correctly decided null)
        if offer is None:
            return 1.0
            
        # If offer exists, must have sources and rule_id
        if offer and sources and isinstance(offer, dict) and offer.get("eligibility_rule_id"):
            return 1.0
            
        return 0.0 # Non-compliant: offer without grounding
    except:
        return 0.0

def evaluate_agent(version=1):
    """
    Runs the LLMOps evaluation pipeline on a specific agent version.
    """
    # Provide the explicit URI to override MLflow Run's local SQLite injection
    os.environ.pop("MLFLOW_RUN_ID", None)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Churn_Prediction_Basic")

    # Load dataset
    eval_df = pd.read_json("data/eval_retention.jsonl", lines=True)
    # Transform to MLflow expected format: 'inputs' column with dict containing the query
    eval_df["inputs"] = eval_df["query"].apply(lambda x: {"query": x})
    eval_df = eval_df.drop(columns=["query", "expected_answer"])
    
    # Initialize Agent
    agent = create_retention_agent(prompt_version=version)

    run_name = f"evaluation_v{version}"
    print(f"Starting deterministic evaluation of {len(eval_df)} queries on local model...")
    
    json_scores = []
    policy_scores = []

    with mlflow.start_run(run_name=run_name):
        mlflow.set_tags({
            "prompt_version": str(version),
            "env": "evaluation"
        })
        
        for idx, row in eval_df.iterrows():
            query = row["inputs"]["query"]
            print(f"\n[{idx+1}/{len(eval_df)}] Query: {query}")
            
            # 1. Invoke Agent
            result = agent.invoke({"input": query})
            output = result["output"]
            
            # 2. Compute explicit programmatic scores
            j_score = json_format_ok(output)
            p_score = discount_policy_compliance(output)
            
            json_scores.append(j_score)
            policy_scores.append(p_score)
            print(f"         JSON Format Score: {j_score} | Policy Compliance Score: {p_score}")
        
        # 3. Aggregate metrics
        metrics = {
            "json_format_ok/mean": sum(json_scores) / len(json_scores) if json_scores else 0.0,
            "discount_policy_compliance/mean": sum(policy_scores) / len(policy_scores) if policy_scores else 0.0,
        }
        
        mlflow.log_metrics(metrics)
        
        print(f"\nEvaluation Metrics for Version {version}:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.2f}")
        
        # Phase 5 Preliminary: Strict Gates (Asserts)
        try:
            assert metrics.get("discount_policy_compliance/mean", 0) == 1.0, "GATE FAIL: Zero tolerance for policy violations!"
            assert metrics.get("json_format_ok/mean", 0) >= 0.80, f"GATE FAIL: JSON format too low ({metrics.get('json_format_ok/mean', 0)})"
            print("\n✅ ALL GATES PASSED for this version.")
        except AssertionError as e:
            print(f"\n❌ GATES FAILED: {str(e)}")
            
        return metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    
    evaluate_agent(version=args.version)
