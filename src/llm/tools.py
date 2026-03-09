import mlflow
import os
import pandas as pd
from langchain_core.tools import tool
from src.llm.search_index import RetentionSearchIndex
from dotenv import load_dotenv

load_dotenv()

# 1. Churn Risk Tool (Queries MLflow Model Registry)
@tool
def get_churn_risk(customer_id: str) -> str:
    """
    Fetches the churn risk prediction for a given customer ID.
    Queries the production-staged 'ChurnModel' from MLflow.
    """
    try:
        # Load production model from Registry
        model_name = "ChurnModel"
        model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        
        # Load customer data
        df = pd.read_csv("data/telco_churn.csv")
        customer_data = df[df['customerID'] == customer_id]
        
        if customer_data.empty:
            return f"Customer {customer_id} not found."
        
        # Preprocess (simplified for the lab)
        from src.churn.loader import load_data
        processed_df = load_data("data/telco_churn.csv")
        row_idx = df[df['customerID'] == customer_id].index[0]
        customer_features = processed_df.iloc[[row_idx]].drop('Churn', axis=1)
        
        # Predict
        prediction = model.predict(customer_features)
        risk_score = float(prediction[0])
        label = "High Risk" if risk_score > 0.5 else "Low Risk"
        
        return f"Churn Risk for {customer_id}: {risk_score:.2f} ({label})."
    except Exception as e:
        return f"Error retrieving churn risk: {str(e)}"

# 2. Policy Retriever Tool (Queries ChromaDB)
@tool
def retrieve_retention_rules(query: str) -> str:
    """
    Searches the corporate retention policy knowledge base (ChromaDB) 
    for applicable discounts or strategies.
    """
    try:
        search_engine = RetentionSearchIndex()
        results = search_engine.search_policy(query)
        if not results:
            return "No specific retention policies found for this customer profile."
        
        formatted_results = "\n".join([
            f"- [RULE_{r['id']}] {r['category']}: {r['benefit']} (Eligibility: {r['condition']})"
            for r in results
        ])
        return f"Relevant Retention Policies:\n{formatted_results}"
    except Exception as e:
        return f"Error searching rules: {str(e)}"

# 3. Escalation Tool (Fallback)
@tool
def escalate_to_human(reason: str) -> str:
    """
    Escalates the case to a human supervisor when no automated policy is found 
    or when technical issues are detected.
    """
    return f"CASE ESCALATED: {reason}. A human agent will take over of this retention case."
