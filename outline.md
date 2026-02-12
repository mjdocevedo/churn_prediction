| Concept MLflow | Module 1   | Module 2        | Module 3       |
| -------------- | ---------- | --------------- | -------------- |
| Run            | Expérience | Pipeline        | Requête        |
| Artefact       | Plot       | Docker / modèle | Prompt / trace |
| Version        | Hyperparam | Model Registry  | Prompt / agent |
| Comparaison    | Modèles    | Versions        | Comportements  |

# Module 1:  MLflow – Introduction (pour tous, surtout Data Scientists)
Objectif : compréhension, rigueur expérimentale, comparaison de modèles.

Contenu :
- Tracking des expériences
- Logging des métriques
- Comparaison de runs
- Lecture et analyse de l’UI


## Outline
1. NB1 
    - Présenter MLflow globalement
    - Vidéo Jerem
    - L'archi Mlflow : Tracking, Projects, Models, Model Registry.
        - Sur le usecase on approfondit Mlflow Tracking (parametres, metriques, plots) et MLflow Models (partiellement, comprendre le concept de l'artifact Logged Model, comprendre que MLflow sauvegarde le model pour pouvoir y revenir plus tard)
2. NB2
    - Clone du repo, présentation de l'achi minimale à faire évoluer
        ```bash
            churn_prediction/
            ├── .venv/
            ├── data/
            │   └── telco_churn.csv
            ├── src/
            │   ├── evaluate.py
            │   ├── loader.py
            │   ├── serve.py
            │   └── train.py
            ├── .gitignore
            ├── Dockerfile
            ├── MLproject
            ├── pyproject.toml
            └── README.md
        ```
    - Intro UV brievement, mise en place à travers pyproject.toml.
    - Modifier `src/train.py` pour mettre en place MLFlow, puis lancer le script `uv run src/train.py`. À evoquer : 
        * Qu'est-ce que c'est `mlruns` ? -> Artifact Store. Sauvegarde les fichiers (models, plots, requirements.txt) pour chaque run. C'est comme le "hard drive" des issus des expériences. À noter qu'en prod on le remplace par un S3/Blob/autre. (Approfondir dans le Module 2)
        * Qu'est ce que c'est `mlflow.db`? -> Backend Store (une base de données SQLite). Sauvegarde la metadonnée (metriques, paramètres, nom des runs, tags). C'est comme l'indexe ou catalogue dont MLflow UI a besoin pour afficher les tableaux de runs.  À noter qu'en prod on le remplace par PostgreSQL (Approfondir dans le Module 2)
    - Accéder au MLFlow UI à travers la commande `uv run mlflow ui`
        * Expliquer que le port 5000 est le port défini par défaut dans le module MLflow. Sauf explicité autrement, 127.0.0.1:5000.
        * Est-ce modifiable ? 
            1. Pour changer le port port: `-p` or `--port`. `uv run mlflow ui -p 5001`
            2. Pour changer le hôte: `-h` or `--host`. `uv run mlflow ui -h 0.0.0.0`
    - Une fois dans l'UI
        * Cliquer dans le Training_Run
        * Focus : montrer les paramètres, la configuration utilisée.
        * On a réussi à capturer les expériences, leur définition et un modèle candidat.
    - Modifier `src/evaluate.py` pour mettre en place mlflow et faire une analyse de performance. Puis lancer le script `uv run src/evaluate.py`
        * Un nouveau run, cliquer dans Evaluate_Run.
        * Focus : approfondir sur les artifacts.
        * Des preuves visuelles, matrice de confusion, courbes. MLflow genère des rapports de manière automatique.

Key takeaway: training about creating the object, evaluation about understainding it.


# Module 2: MLflow – Parcours MLOps (spécialisation)
Objectif : industrialisation complète du cycle de vie ML.

Contenu :
- MLflow Projects
- Intégration Docker
- Registry de modèles
- Promotion des versions
- Déploiement (API / serving)
- Interaction avec pipelines

Format :
- Module plus avancé, orienté production
- Examen spécialité :
- 1 repo fourni avec un pipeline existant
- Les apprenants doivent exécuter des commandes MLflow, écrire de petits scripts d’analyse, retrouver des informations dans les runs / registry, puis répondre à un QCM basé sur leurs résultats

# Module 3: MLflow – Parcours LLMOps (spécialisation)
Objectif : observabilité, traçage et évaluation de systèmes LLM / RAG / agents.

Contenu :
- Tracing MLflow
- Evaluation GenAI
- Prompt management
- Analyse de conversations
- Debug d’agents

Format :
- Module orienté systèmes cognitifs
- Examen spécialité :
- 1 repo fourni avec un agent + RAG + traces existantes
- Les apprenants doivent analyser les traces, comprendre les erreurs, investiguer les runs, et répondre à un QCM basé sur leur analyse




# Projet 
### Customer churn prediction 
Dataset : Telco Customer Churn (IBM)
* CSV public
* ~7k lignes
* Classification binaire
* Colonnes simples :
    * numériques
    * catégorielles
* Label : Churn

#### MODULE 1 – MLflow Introduction
Modèle
* Logistic Regression
* RandomForest (optionnel)

MLflow
* Tracking des expériences
* Logging :
    * params (C, max_depth)
    * metrics (accuracy, roc_auc)
    * artefacts (ROC curve, confusion matrix)
📌 Très simple à comprendre
📌 Aucun NLP, aucune feature complexe

#### MODULE 2 – MLOps
Industrialisation naturelle
* MLflow Projects
* Docker
* Registry
* Promotion Staging → Production
* Serving REST

Pipeline
train → evaluate → register → serve

#### MODULE 3 – LLMOps (optionnel / isolé)
Mini use case LLMOps
* Analyse automatique de commentaires clients (autre CSV)
* LLM = outil d’observabilité / analyse
* MLflow utilisé pour :
    * tracing
    * évaluation
    * prompt versioning
Aucun lien direct requis avec le churn.
