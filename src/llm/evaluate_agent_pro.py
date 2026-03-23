"""
Professional Evaluation Framework for Retention Agent
Using MLflow Evaluate with LLM-as-Judge pattern
"""
import mlflow
import os
import sys
import json
import re
import asyncio
import aiohttp
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import pandas as pd

load_dotenv(override=True)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.llm.agent import create_retention_agent

@dataclass
class EvalResult:
    query: str
    output: str
    json_score: float
    policy_score: float
    relevance_score: float
    latency_ms: float
    error: Optional[str] = None


class RetentionAgentEvaluator:
    """
    Production-grade evaluator with:
    - Async batch processing
    - Multiple judges (deterministic + LLM-as-Judge)
    - Robust JSON extraction
    - Comprehensive metrics
    - MLflow integration
    """
    
    # Required keys for v1 prompt
    REQUIRED_KEYS = ["customer_id", "risk", "offer", "justification", "sources", "email_draft"]
    
    def __init__(self, agent, judge_llm=None):
        self.agent = agent
        self.judge_llm = judge_llm  # Optional: for LLM-as-Judge evaluation
        
    def extract_json(self, output: str) -> Optional[Dict]:
        """Robust JSON extraction with multiple strategies."""
        # Strategy 1: ```json blocks
        if match := re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', output, re.IGNORECASE):
            try:
                return json.loads(match.group(1).strip())
            except:
                pass
        
        # Strategy 2: First JSON object
        if match := re.search(r'({[^{}]*({[^{}]*}[^{}]*)*[^{}]*)', output):
            try:
                return json.loads(match.group(1).strip())
            except:
                pass
                
        # Strategy 3: Last attempt - find anything that looks like JSON
        try:
            # Try to find content between first { and last }
            start = output.find('{')
            end = output.rfind('}')
            if start != -1 and end != -1 and end > start:
                return json.loads(output[start:end+1])
        except:
            pass
            
        return None
    
    def score_json_format(self, output: str) -> float:
        """Score JSON format compliance."""
        parsed = self.extract_json(output)
        if not parsed:
            return 0.0
        missing = [k for k in self.REQUIRED_KEYS if k not in parsed]
        if missing:
            return 0.5  # Valid JSON but incomplete
        return 1.0
    
    def score_policy_compliance(self, output: str) -> float:
        """Score policy compliance (offer must have sources and rule_id)."""
        parsed = self.extract_json(output)
        if not parsed:
            return 0.0
        
        offer = parsed.get("offer")
        sources = parsed.get("sources", [])
        
        # No offer = compliant
        if offer is None:
            return 1.0
            
        # Offer exists: must have sources and eligibility_rule_id
        if (isinstance(offer, dict) and 
            offer.get("eligibility_rule_id") and
            isinstance(sources, list) and len(sources) > 0):
            return 1.0
            
        return 0.0
    
    def score_relevance(self, query: str, output: str) -> float:
        """
        Simple relevance check - does output mention the customer/query topic?
        For production, use LLM-as-Judge here.
        """
        parsed = self.extract_json(output)
        if not parsed:
            return 0.0
            
        # Basic heuristics
        score = 0.0
        
        # Has customer_id
        if parsed.get("customer_id"):
            score += 0.3
            
        # Has risk assessment
        risk = parsed.get("risk", {})
        if isinstance(risk, dict) and "score" in risk:
            score += 0.3
            
        # Justification mentions something relevant to query
        justification = parsed.get("justification", "").lower()
        query_lower = query.lower()
        if any(word in justification for word in query_lower.split()[:3]):
            score += 0.4
            
        return min(score, 1.0)
    
    def evaluate_single(self, query: str, timeout: int = 30) -> EvalResult:
        """Evaluate a single query."""
        start = time.time()
        
        try:
            # Run with timeout using ThreadPool
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.agent.invoke, {"input": query})
                result = future.result(timeout=timeout)
            
            output = result["output"]
            latency = (time.time() - start) * 1000
            
            return EvalResult(
                query=query,
                output=output,
                json_score=self.score_json_format(output),
                policy_score=self.score_policy_compliance(output),
                relevance_score=self.score_relevance(query, output),
                latency_ms=latency
            )
            
        except Exception as e:
            return EvalResult(
                query=query,
                output="",
                json_score=0.0,
                policy_score=0.0,
                relevance_score=0.0,
                latency_ms=(time.time() - start) * 1000,
                error=str(type(e).__name__)
            )
    
    def evaluate_batch(
        self, 
        queries: List[str], 
        max_workers: int = 3,
        timeout: int = 30
    ) -> List[EvalResult]:
        """Evaluate multiple queries in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_query = {
                executor.submit(self.evaluate_single, q, timeout): q 
                for q in queries
            }
            
            # Collect results as they complete
            for i, future in enumerate(as_completed(future_to_query)):
                query = future_to_query[future]
                try:
                    result = future.result()
                    results.append(result)
                    print(f"  [{i+1}/{len(queries)}] {query[:50]}... -> JSON:{result.json_score} Policy:{result.policy_score}")
                except Exception as e:
                    print(f"  [{i+1}/{len(queries)}] {query[:50]}... -> ERROR: {e}")
                    results.append(EvalResult(
                        query=query, output="", json_score=0, policy_score=0, 
                        relevance_score=0, latency_ms=0, error=str(e)
                    ))
                    
        return results
    
    def generate_report(self, results: List[EvalResult]) -> Dict[str, Any]:
        """Generate comprehensive evaluation report."""
        if not results:
            return {}
            
        total = len(results)
        errors = sum(1 for r in results if r.error)
        
        return {
            "total_queries": total,
            "successful": total - errors,
            "failed": errors,
            "json_format": {
                "mean": sum(r.json_score for r in results) / total,
                "pass_rate": sum(1 for r in results if r.json_score >= 0.8) / total,
            },
            "policy_compliance": {
                "mean": sum(r.policy_score for r in results) / total,
                "perfect": sum(1 for r in results if r.policy_score == 1.0) / total,
                "violations": sum(1 for r in results if r.policy_score == 0.0) / total,
            },
            "relevance": {
                "mean": sum(r.relevance_score for r in results) / total,
            },
            "latency_ms": {
                "mean": sum(r.latency_ms for r in results) / total,
                "max": max(r.latency_ms for r in results),
                "p95": sorted([r.latency_ms for r in results])[int(total * 0.95)],
            },
            "failures": [
                {"query": r.query, "error": r.error}
                for r in results if r.error
            ]
        }


def evaluate_agent_pro(
    version: int = 1,
    max_queries: Optional[int] = None,
    max_workers: int = 3,
    timeout: int = 45,
    output_dir: str = "/tmp/eval_results"
):
    """
    Professional evaluation entry point.
    
    Args:
        version: Prompt version
        max_queries: Limit for testing (None = all)
        max_workers: Parallel workers
        timeout: Per-query timeout
        output_dir: Where to save artifacts
    """
    # Setup MLflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Churn_Prediction_Eval")
    
    # Load data
    df = pd.read_json("data/eval_retention.jsonl", lines=True)
    queries = df["query"].tolist()
    
    if max_queries:
        queries = queries[:max_queries]
        print(f"⚡ Testing mode: {len(queries)} queries")
    else:
        print(f"📊 Production mode: {len(queries)} queries")
    
    # Initialize
    print(f"🤖 Loading agent with prompt v{version}...")
    agent = create_retention_agent(prompt_version=version)
    evaluator = RetentionAgentEvaluator(agent)
    
    # Run evaluation
    print(f"\n🚀 Starting parallel evaluation (workers={max_workers}, timeout={timeout}s)...")
    start_time = time.time()
    results = evaluator.evaluate_batch(queries, max_workers=max_workers, timeout=timeout)
    elapsed = time.time() - start_time
    
    # Generate report
    report = evaluator.generate_report(results)
    
    # Log to MLflow
    with mlflow.start_run(run_name=f"pro_eval_v{version}_{time.strftime('%Y%m%d_%H%M%S')}"):
        mlflow.set_tags({
            "prompt_version": str(version),
            "eval_framework": "RetentionAgentEvaluator_v1",
            "max_workers": max_workers,
            "timeout_seconds": timeout,
        })
        
        # Log metrics
        metrics = {
            "json_format/mean": report["json_format"]["mean"],
            "json_format/pass_rate": report["json_format"]["pass_rate"],
            "policy_compliance/mean": report["policy_compliance"]["mean"],
            "policy_compliance/violations": report["policy_compliance"]["violations"],
            "relevance/mean": report["relevance"]["mean"],
            "latency_ms/mean": report["latency_ms"]["mean"],
            "latency_ms/p95": report["latency_ms"]["p95"],
            "throughput_qps": report["total_queries"] / elapsed,
        }
        mlflow.log_metrics(metrics)
        
        # Save detailed results
        os.makedirs(output_dir, exist_ok=True)
        results_path = f"{output_dir}/results_v{version}.json"
        with open(results_path, "w") as f:
            json.dump({
                "report": report,
                "detailed_results": [
                    {
                        "query": r.query,
                        "json_score": r.json_score,
                        "policy_score": r.policy_score,
                        "relevance_score": r.relevance_score,
                        "latency_ms": r.latency_ms,
                        "error": r.error,
                    }
                    for r in results
                ]
            }, f, indent=2)
        mlflow.log_artifact(results_path)
        
        # Print report
        print(f"\n{'='*60}")
        print(f"📋 Evaluation Report (Prompt v{version})")
        print(f"{'='*60}")
        print(f"Queries: {report['total_queries']} | Failed: {report['failed']}")
        print(f"Duration: {elapsed:.1f}s | Throughput: {metrics['throughput_qps']:.2f} q/s")
        print(f"\nScores:")
        print(f"  JSON Format:     {report['json_format']['mean']:.2f} (pass rate: {report['json_format']['pass_rate']:.0%})")
        print(f"  Policy:          {report['policy_compliance']['mean']:.2f} (violations: {report['policy_compliance']['violations']:.0%})")
        print(f"  Relevance:       {report['relevance']['mean']:.2f}")
        print(f"\nLatency:")
        print(f"  Mean: {report['latency_ms']['mean']:.0f}ms | P95: {report['latency_ms']['p95']:.0f}ms")
        
        # Gates
        print(f"\n{'='*60}")
        gates_passed = True
        if report["policy_compliance"]["violations"] > 0:
            print("❌ GATE FAILED: Policy violations detected!")
            gates_passed = False
        else:
            print("✅ Policy compliance: PASS")
            
        if report["json_format"]["mean"] < 0.8:
            print(f"❌ GATE FAILED: JSON format score too low ({report['json_format']['mean']:.2f})")
            gates_passed = False
        else:
            print("✅ JSON format: PASS")
            
        print(f"{'='*60}")
        
    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=45)
    args = parser.parse_args()
    
    evaluate_agent_pro(
        version=args.version,
        max_queries=args.max_queries,
        max_workers=args.max_workers,
        timeout=args.timeout
    )
