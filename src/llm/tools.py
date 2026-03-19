import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pandas as pd
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv(override=True)


@tool
def get_churn_risk(customer_id: str) -> str:
    """
    Fetches the churn risk prediction for a given customer ID.
    Queries the production-staged 'ChurnModel' via the MLflow model server.
    """
    import requests
    from churn.loader import load_data

    try:
        # 1. Load the raw CSV to find the row index for this customer
        raw_df = pd.read_csv("data/telco_churn.csv")
        customer_row = raw_df[raw_df["customerID"] == customer_id]

        if customer_row.empty:
            return f"Customer {customer_id} not found in the dataset."

        row_idx = customer_row.index[0]

        # 2. Load the processed feature matrix (same preprocessing as training)
        processed_df = load_data("data/telco_churn.csv")
        customer_features = processed_df.drop("Churn", axis=1).iloc[[row_idx]]

        # 3. Build the payload for MLflow's scoring server
        #    Use orient="split" but strip the "index" key — MLflow doesn't expect it.
        split = customer_features.to_dict(orient="split")
        payload = {
            "dataframe_split": {
                "columns": split["columns"],
                "data": split["data"],
            }
        }

        # 4. POST to the model server
        model_server_url = os.getenv(
            "MODEL_SERVER_URL", "http://localhost:5001/invocations"
        )
        response = requests.post(
            model_server_url,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        response.raise_for_status()

        # 5. Parse prediction
        result = response.json()
        predictions = result.get("predictions", result.get("outputs", []))
        risk_score = float(predictions[0])
        label = "High Risk" if risk_score > 0.5 else "Low Risk"
        return f"Churn Risk for {customer_id}: {risk_score:.2f} ({label})."

    except requests.exceptions.ConnectionError:
        return (
            f"Churn Risk for {customer_id}: N/A "
            f"(Model server unreachable at {os.getenv('MODEL_SERVER_URL', 'http://localhost:5001/invocations')})"
        )
    except requests.exceptions.HTTPError as e:
        return f"Churn Risk for {customer_id}: N/A (Model server error: {e.response.status_code} - {e.response.text})"
    except Exception as e:
        return f"Churn Risk for {customer_id}: N/A (Error: {type(e).__name__}: {e})"


@tool
def retrieve_retention_rules(query: str) -> str:
    """
    Searches the corporate retention policy knowledge base (ChromaDB)
    for applicable discounts or strategies.
    """
    try:
        from search_index import RetentionSearchIndex
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
        return f"Error searching rules: {type(e).__name__}: {e}"


@tool
def escalate_to_human(reason: str) -> str:
    """
    Escalates the case to a human supervisor when no automated policy is found
    or when technical issues are detected.
    """
    return f"CASE ESCALATED: {reason}. A human agent will take over this retention case."
