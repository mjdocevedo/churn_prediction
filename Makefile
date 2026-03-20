# =============================================================================
#  Makefile — Churn Prediction MLOps Project
# =============================================================================
#
#  STANDARD WORKFLOW
#  -----------------
#  make build             → Build the Docker image (run once, or after code changes)
#  make infra-up          → Start postgres + mlflow-server + chroma
#  make docker-pipeline   → Run full pipeline INSIDE Docker (shares mlruns_data volume)
#  make model-server-up   → Start the model server
#  make health            → Check model server is responding
#  make test-agent        → Run the agent trace test (locally)
#
#  ⚠ NOTE ON LOCAL PIPELINE:
#  `make pipeline` runs the pipeline locally. It connects to the Docker
#  mlflow-server at http://localhost:5000 and uploads artifacts through
#  the proxy. However, the artifacts land INSIDE the Docker volume, not
#  on your host filesystem — so `make docker-pipeline` is the safer choice.
#
#  TEARDOWN:
#  make infra-down        → Stop containers (keeps data)
#  make reset             → ⚠ Stop containers AND delete all volumes (full clean)
# =============================================================================

COMPOSE_FILE   := docker/compose.yml
MLFLOW_URI     := http://localhost:5000

# ---- Build ----------------------------------------------------------------

.PHONY: build
build: ## Build the shared Docker image (churn-prediction-env)
	docker-compose -f $(COMPOSE_FILE) build

# ---- Infrastructure -------------------------------------------------------

.PHONY: infra-up
infra-up: ## Start postgres, mlflow-server, and chroma-server
	docker-compose -f $(COMPOSE_FILE) up -d postgres mlflow-server chroma-server
	@echo ""
	@echo "  MLflow UI: http://localhost:5000"
	@echo "  ChromaDB:  http://localhost:8000"
	@echo "  (Wait ~5s for mlflow-server to be ready before running the pipeline)"

.PHONY: infra-down
infra-down: ## Stop all containers (data volumes are preserved)
	docker-compose -f $(COMPOSE_FILE) down

.PHONY: infra-logs
infra-logs: ## Tail logs for infrastructure containers
	docker-compose -f $(COMPOSE_FILE) logs -f mlflow-server postgres

# ---- Pipeline (inside Docker — RECOMMENDED) --------------------------------

.PHONY: docker-pipeline
docker-pipeline: ## Run the full MLOps pipeline inside Docker (artifacts go to shared volume)
	docker-compose -f $(COMPOSE_FILE) run --rm pipeline-runner
	@echo ""
	@echo "  Pipeline complete. Run 'make model-server-up' to serve the model."

# ---- Pipeline (local — connects to Docker mlflow-server) -------------------

.PHONY: pipeline
pipeline: ## Run the full pipeline locally (connects to Docker mlflow-server)
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) uv run src/pipeline.py

.PHONY: train
train: ## Run training step only (local)
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) uv run src/churn/train.py

.PHONY: evaluate
evaluate: ## Run evaluation step only (local)
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) uv run src/churn/evaluate.py

.PHONY: register
register: ## Run registration step only (local)
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) uv run src/churn/register.py

.PHONY: promote
promote: ## Run promotion step only (local)
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) uv run src/churn/promote.py

# ---- Model Server ----------------------------------------------------------

.PHONY: model-server-up
model-server-up: ## Start the model server (run after docker-pipeline)
	docker-compose -f $(COMPOSE_FILE) up -d model-server
	@echo "  Model server starting at http://localhost:5001"
	@echo "  Run 'make model-server-logs' to monitor startup."

.PHONY: model-server-logs
model-server-logs: ## Tail model-server logs
	docker-compose -f $(COMPOSE_FILE) logs -f model-server

.PHONY: model-server-down
model-server-down: ## Stop the model server
	docker-compose -f $(COMPOSE_FILE) stop model-server

# ---- LLMOps & Agent --------------------------------------------------------

.PHONY: init-search-index
init-search-index: ## Initialize the ChromaDB search index (Phase 1)
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) uv run mlflow run . -e init_search_index --env-manager local

.PHONY: register-prompts
register-prompts: ## Register prompts in the MLflow Prompt Registry (Phase 3)
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) uv run mlflow run . -e register_prompts --env-manager local

.PHONY: test-agent
test-agent: ## Run the agent trace test (locally)
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) uv run src/llm/test_agent_trace.py

.PHONY: evaluate-agent
evaluate-agent: ## Evaluate the agent (Phase 4). Usage: make evaluate-agent version=1
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) uv run mlflow run . -e evaluate_agent -P version=$(version) --env-manager local

.PHONY: release-decision
release-decision: ## Run release decision (Phase 5). Usage: make release-decision baseline=1 candidate=2
	MLFLOW_TRACKING_URI=$(MLFLOW_URI) uv run mlflow run . -e release_decision -P baseline=$(baseline) -P candidate=$(candidate) --env-manager local

.PHONY: agent-up
agent-up: ## Start the agent service container
	docker-compose -f $(COMPOSE_FILE) up -d agent-service

# ---- Diagnostics -----------------------------------------------------------

.PHONY: ps
ps: ## Show running containers and their status
	docker-compose -f $(COMPOSE_FILE) ps

.PHONY: health
health: ## Check if the model server is healthy
	@curl -sf http://localhost:5001/health && echo " OK" || echo " model-server not reachable"

.PHONY: mlflow-ui
mlflow-ui: ## Open the MLflow UI in the default browser
	start http://localhost:5000

# ---- Cleanup ---------------------------------------------------------------

.PHONY: reset
reset: ## ⚠ Stop containers AND delete ALL data volumes (full clean slate)
	docker-compose -f $(COMPOSE_FILE) down -v
	@echo ""
	@echo "  All containers stopped and volumes deleted."
	@echo "  Run 'make infra-up && make docker-pipeline && make model-server-up' to start fresh."

# ---- Help ------------------------------------------------------------------

.PHONY: help
help: ## Show available commands
	@echo ""
	@echo "  Churn Prediction MLOps — Commands"
	@echo "  =================================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@echo ""

.DEFAULT_GOAL := help