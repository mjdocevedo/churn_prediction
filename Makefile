# reset local mlruns from scratch
.PHONY: all train evaluate register promote workflow build-model-image build-project-image fix-mlflow-perms clean-broken-mlruns mlflow-sqlite-build-image mlflow-sqlite-up mlflow-sqlite-logs mlflow-sqlite-down train-sqlite evaluate-sqlite register-sqlite promote-sqlite workflow-sqlite build-model-image-sqlite reset-mlruns

MLFLOW_EXPERIMENT_NAME := Churn_Prediction_Basic
PY_WARNINGS := ignore::FutureWarning
MLFLOW_SQLITE_URI := http://127.0.0.1:5001
MLFLOW_FILE_URI := file://$(PWD)/mlruns

all: train

clean-broken-mlruns:
	rm -rf mlruns/1

reset-mlruns:
	sudo chown -R $(USER):$(USER) mlruns >/dev/null 2>&1 || true
	rm -rf mlruns
	mkdir -p mlruns

train:
	env -u MLFLOW_TRACKING_URI -u MLFLOW_REGISTRY_URI PYTHONWARNINGS=$(PY_WARNINGS) MLFLOW_TRACKING_URI=$(MLFLOW_FILE_URI) MLFLOW_REGISTRY_URI=$(MLFLOW_FILE_URI) MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) uv run mlflow run . -e train

evaluate:
	env -u MLFLOW_TRACKING_URI -u MLFLOW_REGISTRY_URI PYTHONWARNINGS=$(PY_WARNINGS) MLFLOW_TRACKING_URI=$(MLFLOW_FILE_URI) MLFLOW_REGISTRY_URI=$(MLFLOW_FILE_URI) MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) uv run mlflow run . -e evaluate

register:
	env -u MLFLOW_TRACKING_URI -u MLFLOW_REGISTRY_URI PYTHONWARNINGS=$(PY_WARNINGS) MLFLOW_TRACKING_URI=$(MLFLOW_FILE_URI) MLFLOW_REGISTRY_URI=$(MLFLOW_FILE_URI) MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) uv run mlflow run . -e register

promote:
	env -u MLFLOW_TRACKING_URI -u MLFLOW_REGISTRY_URI PYTHONWARNINGS=$(PY_WARNINGS) MLFLOW_TRACKING_URI=$(MLFLOW_FILE_URI) MLFLOW_REGISTRY_URI=$(MLFLOW_FILE_URI) MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) uv run mlflow run . -e promote

workflow:
	env -u MLFLOW_TRACKING_URI -u MLFLOW_REGISTRY_URI PYTHONWARNINGS=$(PY_WARNINGS) MLFLOW_TRACKING_URI=$(MLFLOW_FILE_URI) MLFLOW_REGISTRY_URI=$(MLFLOW_FILE_URI) MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) uv run mlflow run . -e workflow

build-model-image:
	env -u MLFLOW_TRACKING_URI -u MLFLOW_REGISTRY_URI PYTHONWARNINGS=$(PY_WARNINGS) MLFLOW_TRACKING_URI=$(MLFLOW_FILE_URI) MLFLOW_REGISTRY_URI=$(MLFLOW_FILE_URI) MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) uv run src/churn/build_model_image.py

build-project-image:
	docker build -t churn-prediction-project-env -f docker/Dockerfile.project .

fix-mlflow-perms:
	sudo chown -R $(USER):$(USER) mlruns

# --- Clean SQLite tracking workflow (recommended for registry) ---
mlflow-sqlite-build-image:
	docker build -t mlflow-sqlite-server -f docker/Dockerfile.mlflow-sqlite .

mlflow-sqlite-up:
	docker rm -f mlflow-sqlite >/dev/null 2>&1 || true
	docker run -d --name mlflow-sqlite -p 5001:5001 -v $(PWD)/mlflow_sqlite:/mlflow mlflow-sqlite-server

mlflow-sqlite-logs:
	docker logs mlflow-sqlite

mlflow-sqlite-down:
	docker rm -f mlflow-sqlite

train-sqlite:
	MLFLOW_TRACKING_URI=$(MLFLOW_SQLITE_URI) MLFLOW_REGISTRY_URI=$(MLFLOW_SQLITE_URI) MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) uv run mlflow run . -e train -A network=host

evaluate-sqlite:
	MLFLOW_TRACKING_URI=$(MLFLOW_SQLITE_URI) MLFLOW_REGISTRY_URI=$(MLFLOW_SQLITE_URI) MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) uv run mlflow run . -e evaluate -A network=host

register-sqlite:
	MLFLOW_TRACKING_URI=$(MLFLOW_SQLITE_URI) MLFLOW_REGISTRY_URI=$(MLFLOW_SQLITE_URI) MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) uv run mlflow run . -e register -A network=host

promote-sqlite:
	MLFLOW_TRACKING_URI=$(MLFLOW_SQLITE_URI) MLFLOW_REGISTRY_URI=$(MLFLOW_SQLITE_URI) MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) uv run mlflow run . -e promote -A network=host

workflow-sqlite:
	MLFLOW_TRACKING_URI=$(MLFLOW_SQLITE_URI) MLFLOW_REGISTRY_URI=$(MLFLOW_SQLITE_URI) MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) uv run mlflow run . -e workflow -A network=host

build-model-image-sqlite:
	MLFLOW_TRACKING_URI=$(MLFLOW_SQLITE_URI) MLFLOW_REGISTRY_URI=$(MLFLOW_SQLITE_URI) MLFLOW_EXPERIMENT_NAME=$(MLFLOW_EXPERIMENT_NAME) uv run src/churn/build_model_image.py

sqlite-clean-workflow:
	make mlflow-sqlite-build-image && make mlflow-sqlite-up && make build-project-image && make workflow-sqlite