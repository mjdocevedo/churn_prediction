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
