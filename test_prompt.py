import mlflow
import os

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Churn_Prediction_Basic")

PROMPT_NAME = "test-assistant-prompt"

template = """You are an Intelligent Retention Assistant.
Your goal is to propose a retention offer based on customer churn risk and corporate policy.
Customer Input: {{input}}
"""

mlflow.genai.register_prompt(
    name=PROMPT_NAME,
    template=template,
    commit_message="test"
)
print("done")
