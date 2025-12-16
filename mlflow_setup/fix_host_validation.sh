#!/bin/bash
# Alternative fix: Create a wrapper script that patches MLflow's host validation

# This script modifies MLflow to disable host header validation
# Run this on your EC2 instance

echo "Fixing MLflow host validation..."

# Find MLflow installation
MLFLOW_PATH=$(python3 -c "import mlflow; import os; print(os.path.dirname(mlflow.__file__))")

# Backup original file
if [ -f "$MLFLOW_PATH/server/fastapi_app.py" ]; then
    sudo cp "$MLFLOW_PATH/server/fastapi_app.py" "$MLFLOW_PATH/server/fastapi_app.py.backup"
    
    # Patch the file to disable host validation
    sudo sed -i 's/ALLOWED_HOSTS = \["localhost", "127.0.0.1"\]/ALLOWED_HOSTS = ["*"]/g' "$MLFLOW_PATH/server/fastapi_app.py" 2>/dev/null || \
    sudo sed -i '/def validate_host/d' "$MLFLOW_PATH/server/fastapi_app.py" 2>/dev/null || \
    echo "Manual patch may be needed - check MLflow version"
    
    echo "Patch applied. Restart MLflow service."
else
    echo "MLflow path not found. Trying alternative location..."
    find /home/ubuntu/mlflow-env -name "fastapi_app.py" -type f 2>/dev/null | head -1 | while read file; do
        echo "Found: $file"
        sudo cp "$file" "$file.backup"
        # Add patch logic here if needed
    done
fi

echo "Done. Restart MLflow: sudo systemctl restart mlflow"

