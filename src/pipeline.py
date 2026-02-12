"""
pipeline.py — End-to-end MLOps Pipeline

Orchestrates the full workflow:
  Train → Evaluate → Register → Promote

Can be run locally via `uv run src/pipeline.py`
or inside a container via `docker compose run pipeline-runner`.
"""
import mlflow
import sys
import os
import importlib

# Add src/ to the Python path so churn package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from churn.train import train
from churn.evaluate import evaluate
from churn.register import register
from churn.promote import promote


def run_pipeline():
    print("=" * 60)
    print("  MLOps Pipeline: Train → Evaluate → Register → Promote")
    print("=" * 60)

    # 1. Train
    print("\n[Step 1/4] Training Model...")
    run_id = train()
    print(f"✅ Training Complete. Run ID: {run_id}")

    # 2. Evaluate
    print("\n[Step 2/4] Evaluating Model...")
    evaluate(model_uri=f"runs:/{run_id}/model")
    print("✅ Evaluation Complete.")

    # 3. Register
    print("\n[Step 3/4] Registering Model...")
    register(run_id=run_id)
    print("✅ Registration Complete.")

    # 4. Promote
    print("\n[Step 4/4] Promoting Model...")
    promote()
    print("✅ Promotion Complete.")

    print("\n" + "=" * 60)
    print("  Pipeline finished successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("  • View experiments: http://localhost:5001 (MLflow UI)")
    print("  • Serve model:      docker compose -f docker/compose.yml up model-server")
    print("  • Build image:      uv run src/churn/build_model_image.py")


if __name__ == "__main__":
    run_pipeline()
