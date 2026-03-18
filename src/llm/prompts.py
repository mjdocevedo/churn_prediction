import mlflow
import os
from mlflow.entities import RunStatus
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv(override=True)

# In MLflow 3.x, GenAI Prompt Registry is a core feature for versioning LLM applications.

PROMPT_NAME = "retention-assistant-prompt"
EXPERIMENT_NAME = "Churn_Prediction_Basic"

def register_prompts():
    """
    Registers Phase 3 prompts (baseline and candidate) into the MLflow Prompt Registry.
    Prompts are associated with EXPERIMENT_NAME so they appear in the MLflow UI.
    """
    # Always point at the configured tracking server, whether called directly or imported.
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    # Associate prompts with the experiment so they appear in the MLflow UI.
    mlflow.set_experiment(EXPERIMENT_NAME)
    # 1. Baseline v0.1: Minimal, no few-shot
    baseline_template = """You are an authorized Intelligent Retention Assistant for a telecommunications company.
Your goal is to propose a legitimate retention offer based on customer churn risk and our internal corporate policy.

### OUTPUT CONTRACT (JSON STRICT)
You MUST output exactly this JSON format:
{{
  "customer_id": "string",
  "risk": {{ "score": float, "label": "string" }},
  "offer": {{ "name": "string", "value": "string", "eligibility_rule_id": "string" }} | null,
  "justification": "string",
  "email_draft": "string",
  "sources": ["rule_id_1", "rule_id_2"]
}}

### ABSOLUTE RULE
- If no rule applies, set "offer" to null and "sources" to [].
- Only propose an offer if evidence supports it.
- Always use the tools to find information.

Customer Input: {{input}}
"""

    mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=baseline_template,
        commit_message="v0.1: Baseline - Strict JSON contract, no few-shot."
    )

    candidate_template = """Analyze the customer input against the provided data and return a JSON object containing the applicable retention offer.

### OUTPUT CONTRACT (JSON STRICT)
{{
  "customer_id": "string",
  "risk": {{ "score": float, "label": "string" }},
  "offer": {{ "name": "string", "value": "string", "eligibility_rule_id": "string" }} | null,
  "justification": "string",
  "sources": ["rule_id_1", "rule_id_2"]
}}

### RULE
- If no rule applies, set "offer" to null and "sources" to [].
- Use only the provided data from retrieve_retention_rules to determine the offer.

### EXAMPLE
Input: "I've been here 3 years and want a discount."
Assistant: {{
  "customer_id": "123",
  "risk": {{ "score": 0.82, "label": "High Risk" }},
  "offer": {{ "name": "Loyalty Discount", "value": "20% off", "eligibility_rule_id": "RULE_policy_loyalty_discount" }},
  "justification": "Customer has 36 months tenure, eligible for discount.",
  "sources": ["RULE_policy_loyalty_discount"]
}}

Customer Input: {{input}}
"""

    mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=candidate_template,
        commit_message="v0.2: Candidate - Added few-shot examples and strict grounding."
    )

def load_prompt_version(version: int):
    """
    Loads a specific version of a prompt from the registry.
    """
    # Note: MLflow Prompt Registry uses explicit integer versions.
    # name_or_uri is a positional argument in mlflow.genai.load_prompt
    return mlflow.genai.load_prompt(PROMPT_NAME, version=version)

if __name__ == "__main__":
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    print("Registering retention assistant prompts...")
    register_prompts()
    print("Prompts successfully registered in MLflow.")
