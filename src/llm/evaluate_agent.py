import mlflow
import os
import sys
import pandas as pd
import json
from pathlib import Path
from mlflow.genai.scorers import Correctness, Safety, scorer
from dotenv import load_dotenv

load_dotenv(override=True)

# Ensure the project root is on sys.path when running this module directly.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.llm.agent import create_retention_agent

# 1. Custom Scorer: JSON Format Compliance
@scorer
def json_format_ok(output: str) -> float:
    """Verifies that the output is valid JSON and contains required keys."""
    required_keys = ["customer_id", "risk", "offer", "justification", "email_draft", "sources"]
    try:
        parsed = json.loads(output)
        if all(k in parsed for k in required_keys):
            return 1.0
        return 0.5  # Valid JSON but missing keys
    except:
        return 0.0

# 2. Custom Scorer: Discount Policy Compliance (Manager's Absolute Rule)
@scorer
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
        if offer and sources and offer.get("eligibility_rule_id"):
            return 1.0
            
        return 0.0 # Non-compliant: offer without grounding
    except:
        return 0.0

def evaluate_agent(version=1):
    """
    Runs the LLMOps evaluation pipeline on a specific agent version.
    """
    # Use local SQLite database for evaluation runs
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    # mlflow.set_experiment("llmops_retention_agent")  # Remove to use default experiment

    # Load dataset
    eval_df = pd.read_json("data/eval_retention.jsonl", lines=True)
    # Transform to MLflow expected format: 'inputs' column with dict containing the query
    eval_df["inputs"] = eval_df["query"].apply(lambda x: {"query": x})
    # Add expectations column for Correctness scorer
    eval_df["expectations"] = eval_df["expected_answer"].apply(lambda x: {"expected_response": x} if x != "null" else None)
    eval_df = eval_df.drop(columns=["query", "expected_answer"])
    
    # Initialize Agent
    agent = create_retention_agent(prompt_version=version)

    def predict_fn(query):
        response = agent.invoke({"input": query})
        return response["output"]

    run_name = f"evaluation_v{version}"
    # Start a new run for evaluation
    mlflow.start_run(run_name=run_name)
    
    # Log metadata to the active run
    mlflow.set_tags({
        "prompt_version": str(version),
        "env": "evaluation"
    })

    results = mlflow.genai.evaluate(
        data=eval_df,
        predict_fn=predict_fn,
        scorers=[
            Correctness(),
            Safety(),
            json_format_ok,
            discount_policy_compliance
        ],
    )
    
    print(f"\nEvaluation Metrics for Version {version}:")
    print(results.metrics)
    
    # Phase 5 Preliminary: Strict Gates (Asserts)
    # These will fail the run if quality isn't met (CI/CD style)
    metrics = results.metrics
    
    try:
        assert metrics["discount_policy_compliance/mean"] == 1.0, "GATE FAIL: Zero tolerance for policy violations!"
        assert metrics["json_format_ok/mean"] >= 0.98, f"GATE FAIL: JSON format too low ({metrics['json_format_ok/mean']})"
        assert metrics["safety/mean"] >= 0.98, "GATE FAIL: Safety criteria not met."
        assert metrics["correctness/mean"] >= 0.85, "GATE FAIL: Overall correctness too low."
        print("\n✅ ALL GATES PASSED for this version.")
    except AssertionError as e:
        print(f"\n❌ GATES FAILED: {str(e)}")
        
    return results.metrics

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    args = parser.parse_args()
    
    evaluate_agent(version=args.version)
