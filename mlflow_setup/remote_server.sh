#!/bin/bash

# MLflow Remote Server Setup Script for AWS EC2
# This script sets up MLflow tracking server with PostgreSQL backend and S3 artifact store

set -e

echo "=========================================="
echo "MLflow Remote Server Setup"
echo "=========================================="

# Configuration variables (update these)
export MLFLOW_BACKEND_STORE_URI="postgresql://username:password@your-neon-postgres-host:5432/mlflowdb"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="s3://your-mlflow-artifacts-bucket/mlflow"
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export AWS_DEFAULT_REGION="us-east-1"

# Install required packages
echo "Installing required packages..."
sudo apt-get update
sudo apt-get install -y python3-pip postgresql-client

# Install MLflow and dependencies
pip3 install mlflow psycopg2-binary boto3

# Create MLflow directory
mkdir -p ~/mlflow-server
cd ~/mlflow-server

# Start MLflow server
echo "Starting MLflow server..."
echo "Backend Store: $MLFLOW_BACKEND_STORE_URI"
echo "Artifact Root: $MLFLOW_DEFAULT_ARTIFACT_ROOT"

mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port 5000

echo "MLflow server started on port 5000"
echo "Access it at: http://your-ec2-public-ip:5000"

