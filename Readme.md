# Healthcare ML Risk System (Production-Style)

## Goal
Build an end-to-end machine learning system that predicts patient risk from EHR-like data and runs as a cloud-deployed service.

## What “done” means
- Data ingestion + validation pipeline
- Feature engineering (versioned)
- Model training + evaluation
- Containerized deployment
- Cloud API serving (GCP)
- Monitoring, drift detection, retraining

## Healthcare constraints
- Class imbalance
- Minimize false negatives (missed risk)
- Auditability + safe logging

## Data flow
raw CSV -> scripts/load_data.py -> data/processed/clean.csv
clean.csv -> scripts/build_features.py -> data/processed/features_v1.csv


