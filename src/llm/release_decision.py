import mlflow
import os
from dotenv import load_dotenv

load_dotenv(override=True)

# ── Prompt registry name (must match prompts.py) ──────────────────────────────
PROMPT_NAME = "retention-assistant-prompt"

# ── MLflow Prompt Registry aliases ────────────────────────────────────────────
# `production` → the prompt version currently serving live traffic.
# `challenger`  → the most recent candidate that was evaluated (whether shipped or not).
#
# These aliases allow agent.py (or any serving layer) to load the production
# prompt simply with:
#   mlflow.genai.load_prompt(PROMPT_NAME, version="@production")
ALIAS_PRODUCTION = "production"
ALIAS_CHALLENGER = "challenger"

# ── Evaluation gates ──────────────────────────────────────────────────────────
# All thresholds must stay in sync with the inline gates in evaluate_agent.py.
ABSOLUTE_GATES = {
    "discount_policy_compliance/mean": 1.0,
    "json_format_ok/mean": 0.80,
    "business_relevance/mean": 0.80,
    "risk_grounding/mean": 0.60,
    "llm_judge_business/mean": 0.70,
}

# Candidate must not regress more than 5 % vs baseline on these metrics.
REGRESSION_GUARD_METRICS = [
    "json_format_ok/mean",
    "business_relevance/mean",
    "risk_grounding/mean",
    "llm_judge_business/mean",
]


def get_latest_eval_run(prompt_version: int):
    """Fetches the most recent evaluation run for a specific prompt version."""
    experiment = mlflow.get_experiment_by_name("Churn_Prediction_Basic")
    if experiment is None:
        return None
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.prompt_version = '{prompt_version}'",
        order_by=["start_time DESC"],
        max_results=1,
    )
    return runs.iloc[0] if not runs.empty else None


def _set_prompt_release_metadata(alias: str, version: int, is_shipped: bool):
    """
    Points a registered-prompt alias at a specific version and adds status tags.
    Uses the prompt-specific GenAI API for direct version manipulation.
    """
    try:
        # 1. Set the alias (e.g., @production, @challenger)
        mlflow.genai.set_prompt_alias(
            name=PROMPT_NAME,
            alias=alias,
            version=version
        )
        print(f"MLflow alias '{alias}' → {PROMPT_NAME} v{version}")
        
        # 2. Add version-level metadata via tags
        # Note: GenAI tags are typically at the model level, but we use this to mark the candidate status.
        status_val = "SHIPPED" if is_shipped else "PROPOSED_ONLY"
        mlflow.genai.set_prompt_tag(
            name=PROMPT_NAME,
            key=f"v{version}_status",
            value=status_val
        )
        mlflow.genai.set_prompt_tag(
            name=PROMPT_NAME,
            key="latest_release_decision",
            value=f"v{version}_{status_val}"
        )
    except Exception as e:
        print(f"Could not update prompt metadata for {alias}: {e}")


def decide_release(baseline_v: int = 1, candidate_v: int = 2):
    """
    Compares baseline vs candidate evaluation runs and makes a release decision.

    On SHIP:
      - Sets the `production` alias to the candidate prompt version.
      - Sets the `challenger` alias to the old baseline (easy rollback reference).

    On NO_SHIP:
      - Keeps the `production` alias on the baseline (no change to live traffic).
      - Sets the `challenger` alias to the candidate (documents the failed attempt).
    """
    os.environ.pop("MLFLOW_RUN_ID", None)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))

    baseline = get_latest_eval_run(baseline_v)
    candidate = get_latest_eval_run(candidate_v)

    if baseline is None or candidate is None:
        print("❌ Error: Missing evaluation runs for comparison.")
        return "NO_SHIP"

    print(f"\n--- RELEASE DECISION: v{baseline_v} (Baseline) vs v{candidate_v} (Candidate) ---")

    b_metrics = baseline.filter(like="metrics.")
    c_metrics = candidate.filter(like="metrics.")

    def get_val(m, name):
        return m.get(f"metrics.{name}", 0)

    issues = []

    # 1. Absolute gates ────────────────────────────────────────────────────────
    print("\n[1/2] Absolute gates:")
    for metric, threshold in ABSOLUTE_GATES.items():
        val = get_val(c_metrics, metric)
        status = "✅" if val >= threshold else "❌"
        print(f"  {status} {metric}: {val:.2f} (required ≥ {threshold})")
        if val < threshold:
            issues.append(f"{metric} = {val:.2f} < {threshold}")

    # 2. Regression guards ────────────────────────────────────────────────────
    print("\n[2/2] Regression guards (≤ 5 % regression allowed):")
    for metric in REGRESSION_GUARD_METRICS:
        b_val = get_val(b_metrics, metric)
        c_val = get_val(c_metrics, metric)
        regressed = c_val < b_val - 0.05001
        status = "❌" if regressed else "✅"
        print(f"  {status} {metric}: baseline={b_val:.2f} → candidate={c_val:.2f}")
        if regressed:
            issues.append(f"Regression in {metric}: {b_val:.2f} → {c_val:.2f}")

    # 3. Decision & alias promotion ───────────────────────────────────────────
    print()
    if issues:
        print("DECISION: ❌ NO_SHIP")
        print("Reason(s):")
        for issue in issues:
            print(f"  - {issue}")
        print("\nUpdating MLflow prompt aliases:")
        _set_prompt_release_metadata(ALIAS_PRODUCTION, baseline_v, is_shipped=False)
        _set_prompt_release_metadata(ALIAS_CHALLENGER, candidate_v, is_shipped=False)
        return "NO_SHIP"
    else:
        print("DECISION: ✅ SHIP")
        print("All gates passed. Promoting candidate to production.")
        print("\nUpdating MLflow prompt aliases:")
        _set_prompt_release_metadata(ALIAS_PRODUCTION, candidate_v, is_shipped=True)
        _set_prompt_release_metadata(ALIAS_CHALLENGER, baseline_v, is_shipped=True)
        return "SHIP"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Compare two evaluated prompt versions and decide whether to promote "
            "the candidate to production. On SHIP, the MLflow Prompt Registry alias "
            f"'{ALIAS_PRODUCTION}' is updated to point at the candidate version, "
            "making it the active prompt for live traffic."
        )
    )
    parser.add_argument("--baseline", type=int, default=1, help="Prompt version used as baseline (default: 1)")
    parser.add_argument("--candidate", type=int, default=2, help="Prompt version being evaluated (default: 2)")
    args = parser.parse_args()

    decide_release(args.baseline, args.candidate)
