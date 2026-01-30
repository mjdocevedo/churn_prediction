FROM python:3.9-slim

# Install uv
RUN pip install uv

# Set working directory
WORKDIR /app

# Copy dependency definition
COPY pyproject.toml uv.lock ./

# Install dependencies using uv
# --system installs into system python, avoiding need for virtualenv activation in container
RUN uv sync --frozen --system

# Copy source code
COPY . .

# Default entrypoint
CMD ["python", "src/train.py"]
