# SQLite + MLproject(Docker) — Clean Setup

## Pourquoi cette variante
Le Model Registry MLflow est recommandé avec un backend base de données. Cette variante utilise un serveur MLflow avec backend SQLite.

## Compatibilité avec MLproject + docker_env
Compatible, à condition que les conteneurs MLproject puissent joindre le serveur MLflow.
Ici on utilise `mlflow run ... -A network=host` et `MLFLOW_TRACKING_URI=http://127.0.0.1:5001`.

## Démarrage
```bash
make mlflow-sqlite-build-image
make mlflow-sqlite-up
make build-project-image
```

## Pipeline complet
```bash
make workflow-sqlite
```

## Build image API du modèle Production
```bash
make build-model-image-sqlite
```

## Arrêt serveur MLflow SQLite
```bash
make mlflow-sqlite-down
```
