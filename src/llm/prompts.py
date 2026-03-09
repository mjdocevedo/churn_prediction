import mlflow
import os
from mlflow.entities import RunStatus
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# In MLflow 3.x, GenAI Prompt Registry is a core feature for versioning LLM applications.

PROMPT_NAME = "retention-assistant-prompt"

def register_prompts():
    """
    Registers Phase 3 prompts (baseline and candidate) into the MLflow Prompt Registry.
    """
    # 1. Baseline v0.1: Minimal, no few-shot
    baseline_template = """You are an Intelligent Retention Assistant.
Your goal is to propose a retention offer based on customer churn risk and corporate policy.

### CONTRAT DE SORTIE (JSON STRICT)
You MUST output exactly this JSON format:
{{
  "customer_id": "string",
  "risk": {{ "score": float, "label": "string" }},
  "offer": {{ "name": "string", "value": "string", "eligibility_rule_id": "string" }} | null,
  "justification": "string",
  "email_draft": "string",
  "sources": ["rule_id_1", "rule_id_2"]
}}

### RÈGLE ABSOLUE
- If no rule applies, set "offer" to null and "sources" to [].
- Never invent a discount.
- Always use the tools to find information.

Customer Input: {{input}}
"""

    mlflow.genai.register_prompt(
        name=PROMPT_NAME,
        template=baseline_template,
        commit_message="v0.1: Baseline - Strict JSON contract, no few-shot."
    )

    # 2. Candidate v0.2: Includes few-shot and improved grounding instructions
    candidate_template = """You are an Intelligent Retention Assistant specializing in customer loyalty.
Your goal is to provide precise, policy-compliant retention offers.

### CONTRAT DE SORTIE (JSON STRICT)
{{
  "customer_id": "string",
  "risk": {{ "score": float, "label": "string" }},
  "offer": {{ "name": "string", "value": "string", "eligibility_rule_id": "string" }} | null,
  "justification": "string",
  "email_draft": "string",
  "sources": ["rule_id_1", "rule_id_2"]
}}

### RÈGLE ABSOLUE
- If no rule applies, set "offer" to null and "sources" to [].
- Never invent a discount. Use only the data from retrieve_retention_rules.
- Precision is critical: rule_id must match the retrieved source.

### EXEMPLE
Input: "I've been here 3 years and want a discount."
Assistant: {{
  "customer_id": "123",
  "risk": {{ "score": 0.82, "label": "High Risk" }},
  "offer": {{ "name": "Loyalty Discount", "value": "20% off", "eligibility_rule_id": "1" }},
  "justification": "Customer has 36 months tenure, eligible for Rule 1.",
  "email_draft": "Thanks for your loyalty! Enjoy 20% off...",
  "sources": ["1"]
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
    return mlflow.genai.load_prompt(name=PROMPT_NAME, version=version)

if __name__ == "__main__":
    mlflow.set_tracking_uri("http://localhost:5000")
    print("Registering retention assistant prompts...")
    register_prompts()
    print("Prompts successfully registered in MLflow.")
