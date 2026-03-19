# Churn Prediction with MLflow

## Project Description
This project demonstrates how to use MLflow to track and manage the machine learning lifecycle for a Customer Churn prediction task. We use the IBM Telco Customer Churn dataset to predict which customers are likely to leave the service.

---

## Module 1: MLflow Introduction 
*Initial experimentation and tracking.*
- **Tracking Experiments**: Log parameters and metrics.
- **Artifact Management**: Save models and visualizations.
- **MLflow UI**: Compare runs and analyze results.

---

## Module 2: MLflow MLOps Integration
**Building upon the tracking foundations of Module 1**, this module focuses on the industrialization and automation of the ML lifecycle.

### Learning Objectives
- **Model Registry**: Centralize model management, versioning, and lifecycle stages (Staging, Production).
- **MLflow Projects**: Standardize environments and entry points using the `MLproject` file and Docker.
- **Automated Governance**: Implement metric-based promotion logic to move models between stages.
- **Model Serving**: Deploy models as REST APIs using MLflow's built-in serving capabilities.
- **Inference Environments**: Package models into Docker images for portable, production-ready inference.

---

## Module 3: MLflow LLMOps & Agent Integration
**Expanding into Generative AI**, this module applies MLflow to build, observe, and industrialize an autonomous LLM Agent (the "Intelligent Retention Assistant").

### Learning Objectives
- **LLM Tracing & Observability**: Use MLflow's `autologging` to trace Agent execution, visualising reasoning spans, tool calls, and payloads.
- **Prompt Registry**: Manage prompt templates professionally by storing, versioning, and fetching them dynamically via the MLflow Prompt Registry.
- **Tool Integration (RAG + Predictive ML)**: Ground the LLM with real data by connecting it to the MLflow Model Server (Predictive Churn) and a local ChromaDB database (Retention Policies).
- **Deterministic LLM Evaluation**: Run automated evaluations with custom programmatic scorers to ensure strict JSON output compliance and absolute policy adherence (zero tolerance for hallucinated discounts).
- **Automated LLMOps Governance**: Compare Candidate vs. Baseline prompt versions to programmatically authorize or block production releases (SHIP / NO_SHIP) based on strict quality gates.
