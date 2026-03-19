import mlflow
import os
from dotenv import load_dotenv

load_dotenv(override=True)

def get_latest_eval_run(prompt_version: int):
    """
    Fetches the most recent evaluation run for a specific prompt version.
    """
    experiment = mlflow.get_experiment_by_name("Churn_Prediction_Basic")
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.prompt_version = '{prompt_version}'",
        order_by=["start_time DESC"],
        max_results=1
    )
    return runs.iloc[0] if not runs.empty else None

def decide_release(baseline_v=1, candidate_v=2):
    """
    Compares baseline vs candidate runs and makes a decision based on strict gates.
    """
    os.environ.pop("MLFLOW_RUN_ID", None)
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    
    baseline = get_latest_eval_run(baseline_v)
    candidate = get_latest_eval_run(candidate_v)

    if baseline is None or candidate is None:
        print("❌ Error: Missing evaluation runs for comparison.")
        return "NO_SHIP"

    print(f"\n--- RELEASE DECISION: v{baseline_v} (Baseline) vs v{candidate_v} (Candidate) ---")
    
    # Extract metrics (prefixed with mean/ for GenAI evaluators)
    b_metrics = baseline.filter(like="metrics.")
    c_metrics = candidate.filter(like="metrics.")

    def get_val(m, name):
        return m.get(f"metrics.{name}", 0)

    # 1. Gates (Absolute Thresholds)
    gates = {
        "discount_policy_compliance/mean": 1.0,
        "json_format_ok/mean": 0.80
    }

    issues = []
    for m, threshold in gates.items():
        val = get_val(c_metrics, m)
        if val < threshold:
            issues.append(f"{m} {val:.2f} < {threshold}")

    # 2. Relative Regressions (No more than 5% regression vs baseline)
    for m in ["json_format_ok/mean"]:
        b_val = get_val(b_metrics, m)
        c_val = get_val(c_metrics, m)
        if c_val < b_val * 0.95:
            issues.append(f"Regression in {m}: {b_val:.2f} -> {c_val:.2f}")

    # 3. Decision
    if issues:
        print("\nDECISION: ❌ NO_SHIP")
        print("Reason(s):")
        for issue in issues:
            print(f"- {issue}")
        return "NO_SHIP"
    else:
        print("\nDECISION: ✅ SHIP")
        print("All gates passed. Metric improvements confirmed.")
        return "SHIP"

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=int, default=1)
    parser.add_argument("--candidate", type=int, default=2)
    args = parser.parse_args()
    
    decide_release(args.baseline, args.candidate)
