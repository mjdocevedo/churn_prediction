FROM python:3.10-slim
LABEL build_version="2.0-fix-gunicorn"

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency definition only
COPY pyproject.toml ./

# Install dependencies from pyproject.toml
RUN uv sync

# Apply MLflow 3.8.1 FastAPI compatibility fix
# Issue: MLflow 3.8.1 uses Flask-style @app.route() decorators on FastAPI app
# Solution: Replace with FastAPI-style @app.get()/@app.post() decorators
RUN MLFLOW_FILE="/app/.venv/lib/python3.10/site-packages/mlflow/pyfunc/scoring_server/__init__.py" && \
    sed -i 's/@app\.route("\/ping", methods=\["GET"\])/@app.get("\/ping")/g' "$MLFLOW_FILE" && \
    sed -i 's/@app\.route("\/health", methods=\["GET"\])/@app.get("\/health")/g' "$MLFLOW_FILE" && \
    sed -i 's/@app\.route("\/version", methods=\["GET"\])/@app.get("\/version")/g' "$MLFLOW_FILE" && \
    sed -i 's/@app\.route("\/invocations", methods=\["POST"\])/@app.post("\/invocations")/g' "$MLFLOW_FILE"

# Add the virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY src/ src/
COPY data/ data/
COPY MLproject ./

# Default entrypoint
CMD ["python", "src/churn/train.py"]
