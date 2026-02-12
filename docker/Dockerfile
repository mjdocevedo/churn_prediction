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

# Add the virtual environment to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY src/ src/
COPY data/ data/
COPY MLproject ./

# Default entrypoint
CMD ["python", "src/churn/train.py"]
