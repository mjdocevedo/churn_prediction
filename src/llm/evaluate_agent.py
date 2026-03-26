import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Optional

import mlflow
import pandas as pd
from dotenv import load_dotenv
from mlflow.genai.scorers import scorer
from openai import OpenAI

load_dotenv(override=False)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.llm.agent import create_retention_agent


def extract_json_from_output(output: str) -> Optional[str]:
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
    matches = re.findall(json_block_pattern, output or "", re.IGNORECASE)
    for match in matches:
        try:
            json.loads(match.strip())
            return match.strip()
        except Exception:
            continue

    json_pattern = r"({[\s\S]*})"
    matches = re.findall(json_pattern, output or "")
    for match in matches:
        try:
            json.loads(match.strip())
            return match.strip()
        except Exception:
            continue
    return None


def _parse_output_json(output: str) -> dict:
    candidate = extract_json_from_output(output) or (output or "")
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _query_has_customer_id(query: str) -> bool:
    return bool(re.search(r"(?:customer\s*)?id\s*(?::|=|is)?\s*([A-Za-z0-9\-]{4,})", query or "", re.IGNORECASE))


def json_format_ok(output: str) -> float:
    required_keys = ["customer_id", "risk", "offer", "justification", "sources", "email_draft"]
    parsed = _parse_output_json(output)
    if not parsed:
        return 0.0
    return 1.0 if all(k in parsed for k in required_keys) else 0.5


def discount_policy_compliance(output: str) -> float:
    parsed = _parse_output_json(output)
    if not parsed:
        return 0.0
    offer = parsed.get("offer")
    sources = parsed.get("sources", [])
    if offer is None:
        return 1.0
    if isinstance(offer, dict) and offer.get("eligibility_rule_id") and isinstance(sources, list) and len(sources) > 0:
        return 1.0
    return 0.0


# Canonical offer keywords from retention_knowledge.json
_OFFER_KEYWORD_MAP = {
    "loyalty discount": ["loyalty discount", "policy_loyalty_discount", "loyalty"],
    "one month free contract upgrade": ["one month free", "policy_contract_upgrade", "contract upgrade", "free month"],
    "streaming service bundle": ["streaming service bundle", "policy_fiber_promo", "streaming bundle", "fiber promo"],
    "fiber optic special offer": ["fiber optic", "policy_fiber_promo", "fiber promo", "streaming"],
    "priority support": ["priority support", "policy_support_escalation", "support ticket"],
}


def business_relevance_score(output: str, expected_answer: str) -> float:
    parsed = _parse_output_json(output)
    expected = (expected_answer or "").strip().lower()
    offer = parsed.get("offer") if isinstance(parsed, dict) else None
    offer_name = str(offer.get("name", "")).lower() if isinstance(offer, dict) else ""
    text = (extract_json_from_output(output) or output or "").lower()

    # Expected no offer
    if expected in {"null", "none", ""}:
        return 1.0 if offer is None else 0.0

    # Expected a specific named offer — match by keyword
    for offer_key, keywords in _OFFER_KEYWORD_MAP.items():
        if expected == offer_key or any(kw in expected for kw in keywords):
            if not isinstance(offer, dict):
                # Check if the text at least mentions offer-related keywords
                return 0.5 if any(kw in text for kw in keywords) else 0.0
            if not offer.get("eligibility_rule_id"):
                return 0.5
            if any(kw in offer_name for kw in keywords) or any(kw in text for kw in keywords):
                return 1.0
            return 0.0

    # Fallback: expected contains an offer keyword anywhere
    offer_indicators = ["loyalty", "discount", "upgrade", "bundle", "promo", "streaming", "fiber", "support"]
    if any(ind in expected for ind in offer_indicators):
        if not isinstance(offer, dict):
            return 0.0
        if any(ind in offer_name for ind in offer_indicators):
            return 1.0 if offer.get("eligibility_rule_id") else 0.5
        return 0.0

    return 0.0


def customer_id_quality_score(output: str, query: str = "") -> float:
    parsed = _parse_output_json(output)
    cid = str(parsed.get("customer_id", "")).strip().lower() if parsed else ""
    query_has_id = _query_has_customer_id(query)

    if query_has_id:
        return 1.0 if cid and cid not in {"unknown", "not_found", "none", "null", ""} else 0.0

    # neutral score when query doesn't contain a customer id
    return 0.5


def risk_grounding_score(output: str, query: str = "") -> float:
    """Score risk grounding quality.

    - Query has customer ID: agent MUST look up and return a non-trivial risk → 0 or 1.
    - Query has churn signals (cancel, expensive, unhappy): agent should signal medium+ risk → 0.5 or 1.
    - Generic/safety/OOF query: agent should return low/null risk or null offer → 1.0 if risk score ≤ 0.3, else 0.5.
    """
    parsed = _parse_output_json(output)
    risk = parsed.get("risk", {}) if isinstance(parsed, dict) else {}
    risk_score = None
    label = ""
    if isinstance(risk, dict):
        risk_score = risk.get("score")
        label = str(risk.get("label", "")).strip().lower()

    # Queries with explicit customer ID — must produce a real risk lookup
    if _query_has_customer_id(query):
        if risk_score in [None, 0, 0.0] or label in {"unknown", "n/a", "none", "null", ""}:
            return 0.0
        return 1.0

    # Queries with churn signals — expect at least medium risk or escalation
    churn_signals = ["cancel", "unhappy", "too expensive", "leaving", "quit", "terminate"]
    query_lower = (query or "").lower()
    if any(sig in query_lower for sig in churn_signals):
        if risk_score is not None and risk_score >= 0.4:
            return 1.0
        return 0.5  # partial credit — agent replied but under-estimated risk

    # Out-of-scope / safety / general queries — risk should be low or null
    if risk_score is None or risk_score <= 0.3:
        return 1.0
    return 0.5  # agent assigned risk to an unrelated query — mild penalty




def llm_judge_score(query: str, output: str, expected_answer: str) -> float:
    """Use an external judge model (default gpt-4o-mini) to score response quality [0,1]."""
    judge_enabled = os.getenv("JUDGE_ENABLED", "1") == "1"
    if not judge_enabled:
        return 0.5

    api_key = os.getenv("JUDGE_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_KEY")
    if not api_key:
        return 0.0

    judge_model = os.getenv("JUDGE_LLM_MODEL", "gpt-4o-mini")
    judge_base_url = os.getenv("JUDGE_BASE_URL") or os.getenv("LITELLM_BASE_URL", "https://ai-gateway.liora.tech/")

    prompt = (
        "You are an evaluation judge for a retention assistant. "
        "Score the assistant response from 0 to 1. Return ONLY JSON: {\"score\": <float>, \"reason\": \"...\"}.\n"
        "Scoring rules:\n"
        "- 1.0: Correct, policy-grounded, matches expected intent.\n"
        "- 0.5: Partially correct or generic but acceptable.\n"
        "- 0.0: Wrong intent, unsafe policy behavior, or irrelevant.\n\n"
        f"Query: {query}\n"
        f"Expected: {expected_answer}\n"
        f"Assistant output: {output}\n"
    )

    try:
        client = OpenAI(api_key=api_key, base_url=judge_base_url)
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        txt = (resp.choices[0].message.content or "").strip()
        parsed = _parse_output_json(txt)
        if parsed and "score" in parsed:
            score = float(parsed.get("score", 0.0))
            return max(0.0, min(1.0, score))

        # Fallback: extract first float in [0,1] from free-text answer
        m = re.search(r"[01](?:\.\d+)?", txt)
        if m:
            score = float(m.group(1))
            return max(0.0, min(1.0, score))
        return 0.0
    except Exception:
        return 0.0


def evaluate_agent(version=1, max_queries: Optional[int] = None, timeout_seconds: int = 60, debug: bool = True):
    # Agent model selection (separate from evaluator/judge model if needed)
    agent_provider = os.getenv("AGENT_LLM_PROVIDER", os.getenv("LLM_PROVIDER", "openai")).lower()
    agent_model = os.getenv("AGENT_LLM_MODEL", os.getenv("LLM_MODEL", "gpt-4o-mini"))

    os.environ["LLM_PROVIDER"] = agent_provider
    os.environ["LLM_MODEL"] = agent_model

    if agent_provider != "ollama":
        os.environ.setdefault("LITELLM_BASE_URL", "https://ai-gateway.liora.tech/")
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_KEY", "")

    # Judge model config (separate model used only for evaluation scoring)
    os.environ.setdefault("JUDGE_LLM_MODEL", os.getenv("JUDGE_LLM_MODEL", "gpt-4o-mini"))
    os.environ.setdefault("JUDGE_BASE_URL", os.getenv("JUDGE_BASE_URL", os.getenv("LITELLM_BASE_URL", "https://ai-gateway.liora.tech/")))
    os.environ.setdefault("JUDGE_API_KEY", os.getenv("JUDGE_API_KEY", os.getenv("OPENAI_API_KEY") or os.getenv("LITELLM_KEY", "")))

    os.environ.pop("MLFLOW_RUN_ID", None)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Churn_Prediction_Basic")

    # Stability knobs for local eval
    os.environ.setdefault("MLFLOW_GENAI_EVAL_SKIP_TRACE_VALIDATION", "True")
    os.environ.setdefault("MLFLOW_GENAI_EVAL_MAX_WORKERS", "1")

    eval_path = ROOT / "data" / "eval_retention.jsonl"
    eval_df = pd.read_json(eval_path, lines=True)
    if max_queries:
        eval_df = eval_df.head(max_queries)
        print(f"⚡ Fast mode: evaluating only {max_queries} queries")

    print(f"Agent model for evaluation calls: provider={agent_provider}, model={agent_model}")
    print(f"Judge model for scoring: model={os.getenv('JUDGE_LLM_MODEL','gpt-4o-mini')}, base_url={os.getenv('JUDGE_BASE_URL','')}")
    agent = create_retention_agent(prompt_version=version)

    data = pd.DataFrame(
        {
            "inputs": eval_df["query"].apply(lambda x: {"query": x}),
            "expectations": eval_df["expected_answer"].apply(lambda x: {"expected_response": x}),
        }
    )

    stats = {"latencies": [], "errors": 0}

    def predict_fn(query):
        query = str(query)
        if debug:
            idx = len(stats["latencies"]) + stats["errors"] + 1
            print(f"\n[{idx}/{len(data)}] Query: {query}")

        start = time.time()
        try:
            result = agent.invoke({"input": query})
            out = result.get("output", "")
            lat = time.time() - start
            stats["latencies"].append(lat)
            if debug:
                print(f"   [TIME] {lat:.1f}s")
                print(f"   [RAW] {str(out)[:140]}...")
            return out
        except Exception as e:
            stats["errors"] += 1
            if debug:
                print(f"   ❌ ERROR: {type(e).__name__}: {e}")
            return ""

    @scorer(name="json_format_ok", aggregations=["mean"])
    def json_scorer(outputs):
        return json_format_ok(outputs)

    @scorer(name="discount_policy_compliance", aggregations=["mean"])
    def policy_scorer(outputs):
        return discount_policy_compliance(outputs)

    @scorer(name="business_relevance", aggregations=["mean"])
    def business_scorer(outputs, expectations):
        expected = expectations.get("expected_response", "") if isinstance(expectations, dict) else ""
        return business_relevance_score(outputs, expected)

    @scorer(name="customer_id_quality", aggregations=["mean"])
    def customer_id_scorer(outputs, inputs):
        q = inputs.get("query", "") if isinstance(inputs, dict) else str(inputs)
        return customer_id_quality_score(outputs, q)

    @scorer(name="risk_grounding", aggregations=["mean"])
    def risk_grounding_scorer(outputs, inputs):
        q = inputs.get("query", "") if isinstance(inputs, dict) else str(inputs)
        return risk_grounding_score(outputs, q)

    @scorer(name="llm_judge_business", aggregations=["mean"])
    def llm_judge_business_scorer(outputs, inputs, expectations):
        q = inputs.get("query", "") if isinstance(inputs, dict) else str(inputs)
        expected = expectations.get("expected_response", "") if isinstance(expectations, dict) else ""
        return llm_judge_score(q, outputs, expected)

    with mlflow.start_run(run_name=f"evaluation_v{version}"):
        mlflow.set_tags(
            {
                "prompt_version": str(version),
                "env": "evaluation",
                "eval_framework": "mlflow_genai_evaluate",
                "timeout_seconds": timeout_seconds,
                "agent_llm_provider": agent_provider,
                "agent_llm_model": agent_model,
            }
        )

        result = mlflow.genai.evaluate(
            data=data,
            predict_fn=predict_fn,
            scorers=[json_scorer, policy_scorer, business_scorer, customer_id_scorer, risk_grounding_scorer, llm_judge_business_scorer],
        )

        metrics = dict(result.metrics)
        metrics["avg_latency_seconds"] = float(sum(stats["latencies"]) / len(stats["latencies"])) if stats["latencies"] else 0.0
        metrics["failed_queries"] = float(stats["errors"])
        metrics["total_queries"] = float(len(data))
        mlflow.log_metrics(metrics)

        print("\nEvaluation Metrics:")
        for k, v in sorted(metrics.items()):
            print(f"  {k}: {v:.4f}" if isinstance(v, (int, float)) else f"  {k}: {v}")

        json_mean = metrics.get("json_format_ok/mean", metrics.get("json_format_ok", 0.0))
        policy_mean = metrics.get("discount_policy_compliance/mean", metrics.get("discount_policy_compliance", 0.0))
        biz_mean = metrics.get("business_relevance/mean", metrics.get("business_relevance", 0.0))
        risk_mean = metrics.get("risk_grounding/mean", metrics.get("risk_grounding", 0.0))
        judge_mean = metrics.get("llm_judge_business/mean", metrics.get("llm_judge_business", 0.0))

        if policy_mean < 1.0:
            print("\n❌ GATE FAIL: policy compliance < 1.0")
        elif json_mean < 0.80:
            print(f"\n❌ GATE FAIL: json format too low ({json_mean:.2f})")
        elif biz_mean < 0.80:
            print(f"\n❌ GATE FAIL: business relevance too low ({biz_mean:.2f})")
        elif risk_mean < 0.60:
            print(f"\n❌ GATE FAIL: risk grounding too low ({risk_mean:.2f})")
        elif judge_mean < 0.70:
            print(f"\n❌ GATE FAIL: LLM judge score too low ({judge_mean:.2f})")
        else:
            print("\n✅ ALL GATES PASSED")

        return metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--debug", action="store_true", default=True)
    args = parser.parse_args()

    evaluate_agent(
        version=args.version,
        max_queries=args.max_queries,
        timeout_seconds=args.timeout,
        debug=args.debug,
    )
