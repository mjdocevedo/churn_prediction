# MLflow 3.8.1 FastAPI Fix

## Problem

MLflow 3.8.1 introduced FastAPI-based model serving but uses Flask-style `@app.route()` decorators instead of FastAPI's `@app.get()` and `@app.post()` decorators.

**Error you'll see:**
```
AttributeError: 'FastAPI' object has no attribute 'route'
```

## Root Cause

MLflow 3.8.1 (PR #14307, January 2025) switched the model serving backend from Flask to FastAPI, but the decorator syntax in `mlflow/pyfunc/scoring_server/__init__.py` (line 483+) wasn't fully updated. This causes a mismatch: the file creates a FastAPI app but tries to use Flask-style decorators that don't exist on FastAPI objects.

## Solution

The fix is applied **automatically in Docker during build**. Here's how:

### For Docker (Automatic)

The `Dockerfile` applies the fix using `sed` during the build process:

```dockerfile
RUN MLFLOW_FILE="/app/.venv/lib/python3.10/site-packages/mlflow/pyfunc/scoring_server/__init__.py" && \
    sed -i 's/@app\.route("\/ping", methods=\["GET"\])/@app.get("\/ping")/g' "$MLFLOW_FILE" && \
    sed -i 's/@app\.route("\/health", methods=\["GET"\])/@app.get("\/health")/g' "$MLFLOW_FILE" && \
    sed -i 's/@app\.route("\/version", methods=\["GET"\])/@app.get("\/version")/g' "$MLFLOW_FILE" && \
    sed -i 's/@app\.route("\/invocations", methods=\["POST"\])/@app.post("\/invocations")/g' "$MLFLOW_FILE"
```

### For Local Development (Manual)

If you need to test locally with your `.venv`, apply the fix manually after running `uv sync`:

```bash
cd /home/ubuntu/churn_prediction

MLFLOW_FILE=".venv/lib/python3.10/site-packages/mlflow/pyfunc/scoring_server/__init__.py"

sed -i 's/@app\.route("\/ping", methods=\["GET"\])/@app.get("\/ping")/g' "$MLFLOW_FILE"
sed -i 's/@app\.route("\/health", methods=\["GET"\])/@app.get("\/health")/g' "$MLFLOW_FILE"
sed -i 's/@app\.route("\/version", methods=\["GET"\])/@app.get("\/version")/g' "$MLFLOW_FILE"
sed -i 's/@app\.route("\/invocations", methods=\["POST"\])/@app.post("\/invocations")/g' "$MLFLOW_FILE"
```

## What the fix does

It replaces Flask-style decorators with FastAPI-style decorators in the MLflow serving code:

| Line | Before (Flask) | After (FastAPI) |
|---|---|---|
| 483 | `@app.route("/ping", methods=["GET"])` | `@app.get("/ping")` |
| 484 | `@app.route("/health", methods=["GET"])` | `@app.get("/health")` |
| 494 | `@app.route("/version", methods=["GET"])` | `@app.get("/version")` |
| 501 | `@app.route("/invocations", methods=["POST"])` | `@app.post("/invocations")` |

## Verify the fix

After rebuilding your Docker image or applying the manual fix, test the model server:

```bash
make model-server-logs
```

Expected output (no errors, server running):
```
INFO:     Started server process [21]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:5001 (Press CTRL+C to quit)
```

## Timeline

- **MLflow 3.8.0 and earlier**: Flask-based serving
- **MLflow 3.8.1** (Jan 2025): FastAPI introduced but with Flask decorator syntax bug
- **MLflow 3.8.2+**: Should have this fixed (when released)

## Educational Notes

This bug demonstrates:
1. **Framework migration risks**: Switching frameworks requires thorough testing of all decorator syntax
2. **Why automated testing matters**: This would have been caught by integration tests
3. **FastAPI vs Flask differences**: Different decorator syntax for route registration
