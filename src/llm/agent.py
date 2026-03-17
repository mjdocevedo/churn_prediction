import mlflow
import os
import argparse
from dotenv import load_dotenv

load_dotenv(override=True)
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from src.llm.tools import get_churn_risk, retrieve_retention_rules, escalate_to_human
from src.llm.prompts import load_prompt_version

def create_retention_agent(prompt_version=1):
    """
    Initializes the Retention Agent with tools, strict JSON contract, and MLflow Tracing.
    """
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
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        )
    else:
        print(f"Using OpenAI LLM: {model_name}")
        llm = ChatOpenAI(model=model_name, temperature=0)

    # 3. Define Tools
    tools = [get_churn_risk, retrieve_retention_rules, escalate_to_human]

    # 4. Load Prompt from Registry (Phase 3 alignment)
    prompt_obj = load_prompt_version(version=prompt_version)
    system_message = prompt_obj.template.format(input="{input}")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_message),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    # 5. Create Agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True,
        max_iterations=5,
        handle_parsing_errors=True
    )

    return agent_executor

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    args = parser.parse_args()

    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("llmops_retention_agent")

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
