# README Debug — Exécution clean

Ce projet supporte **2 modes distincts**:
1. **Mode `mlruns` (file-store local)**
2. **Mode SQLite (serveur MLflow dédié)**

Les deux modes utilisent des commandes `make` différentes et des fichiers différents.

---

## Mode A — `mlruns` (file-store local)

### Fichiers utilisés
- `MLproject`
- `docker/Dockerfile.project`
- dossier local `mlruns/`
- `Makefile` (targets standard)

### Commande unique recommandée (from scratch)
```bash
make reset-mlruns && make fix-mlflow-perms && make build-project-image && make workflow
```

### Build de l’image API modèle
```bash
make build-model-image
```

### Lancer l’API
```bash
docker run --rm -p 5000:8080 churn-model-production
```

---

## Mode B — SQLite (recommandé pour registry propre)

### Fichiers utilisés
- `docker/Dockerfile.mlflow-sqlite` (serveur MLflow SQLite)
- dossier local `mlflow_sqlite/` (DB + artefacts du serveur)
- `MLproject`
- `docker/Dockerfile.project`
- `Makefile` (targets `*-sqlite`)

### Commande unique recommandée (from scratch SQLite)
```bash
make sqlite-clean-workflow
```

### Build de l’image API modèle (SQLite)
```bash
make build-model-image-sqlite
```

### Lancer l’API
```bash
docker run --rm -p 5000:8080 churn-model-production
```

### Arrêter le serveur MLflow SQLite
```bash
make mlflow-sqlite-down
```

---

## Règle pratique
- Si tu veux rester simple/local: utilise **Mode A (`mlruns`)**.
- Si tu veux un setup registry plus propre: utilise **Mode B (SQLite)**.
- **Ne mélange pas** les deux modes dans la même exécution.
