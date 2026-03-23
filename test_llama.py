import os
from dotenv import load_dotenv
load_dotenv()

print(f"Model configured: {os.getenv('LLM_MODEL')}")
print(f"Provider: {os.getenv('LLM_PROVIDER')}")

from src.llm.agent import create_retention_agent
agent = create_retention_agent()
print("✅ Agent initialized successfully with llama3.2:1b")
