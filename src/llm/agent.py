import mlflow
import os
import re
import argparse
from dotenv import load_dotenv

load_dotenv(override=True)
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.messages import SystemMessage, HumanMessage
from tools import get_churn_risk, retrieve_retention_rules, escalate_to_human
from prompts import load_prompt_version


def _extract_customer_id(user_input: str) -> str | None:
    """Extract a customer ID from the user's free-text query."""
    match = re.search(r"ID\s*[:=]\s*([A-Za-z0-9\-]+)", user_input)
    return match.group(1) if match else None


class RetentionAgent:
    """Lightweight agent wrapper that calls tools explicitly and then prompts an LLM."""

    def __init__(self, llm, system_message: str):
        self.llm = llm
        self.system_message = system_message

    def invoke(self, inputs: dict) -> dict:
        user_input = inputs.get("input", "")

        # 1) Run tools (wrapped in MLflow spans for trace visibility)
        customer_id = _extract_customer_id(user_input)

        with mlflow.start_span("get_churn_risk"):
            churn_info = get_churn_risk.func(customer_id) if customer_id else "Customer ID not found in query."

        with mlflow.start_span("retrieve_retention_rules"):
            rules_info = retrieve_retention_rules.func(user_input)

        # 2) Generate response (optimized: single efficient LLM call)
        with mlflow.start_span("generate"):
            # Build single optimized prompt combining system message + context + user input
            final_prompt = (
                f"{self.system_message}\n\n"
                f"CHURN RISK ANALYSIS:\n{churn_info}\n\n"
                f"APPLICABLE RETENTION POLICIES:\n{rules_info}\n\n"
                f"CUSTOMER REQUEST:\n{user_input}\n\n"
                f"RESPONSE (JSON):\n"
            )

            # Single efficient invoke() call instead of generate()
            llm_result = self.llm.invoke([HumanMessage(content=final_prompt)])
            output_text = llm_result.content

        return {"output": output_text}


def create_retention_agent(prompt_version=1):
    """Initializes a retention agent that uses an LLM + tool outputs."""

    # 1. Enable MLflow Tracing with custom tags
    mlflow.langchain.autolog()

    # 2. Setup LLM based on provider
    provider = os.getenv("LLM_PROVIDER", "openai").lower()
    model_name = os.getenv("LLM_MODEL", "gpt-4o-mini")

    if provider == "ollama":
        print(f"Using Local LLM via Ollama: {model_name}")
        llm = ChatOllama(
            model=model_name,
            temperature=0,
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        )
    else:
        print(f"Using OpenAI LLM: {model_name}")
        llm = ChatOpenAI(model=model_name, temperature=0)

    # 3. Load Prompt from Registry (Phase 3 alignment)
    prompt_obj = load_prompt_version(version=prompt_version)
    system_message = prompt_obj.template

    return RetentionAgent(llm=llm, system_message=system_message)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Churn_Prediction_Basic")

    agent = create_retention_agent()
    
    if args.serve:
        print("Agent service ready (simulation).")
        # Here we would normally start a FastAPI/Flask server
    else:
        test_query = "Customer 7590-VHVEG is complaining. Check their risk and find a policy."
        with mlflow.start_run(run_name="agent_manual_test"):
            # Set mandatory tags for context
            mlflow.set_tags({
                "prompt_version": "v0.1",
                "churn_model_version": "Production",
                "rules_version": "v1.0",
                "env": "dev"
            })
            response = agent.invoke({"input": test_query})
            print(f"\nResponse:\n{response['output']}")
