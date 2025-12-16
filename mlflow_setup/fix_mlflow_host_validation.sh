#!/bin/bash
# Script to fix MLflow host header validation issue
# Run this on your EC2 instance

set -e

echo "Fixing MLflow host header validation..."

# Activate virtual environment
source ~/mlflow-env/bin/activate

# Find MLflow installation
MLFLOW_PATH=$(python3 -c "import mlflow; import os; print(os.path.dirname(mlflow.__file__))")
echo "MLflow path: $MLFLOW_PATH"

# Find the FastAPI app file
FASTAPI_APP="$MLFLOW_PATH/server/fastapi_app.py"

if [ ! -f "$FASTAPI_APP" ]; then
    echo "Error: Could not find $FASTAPI_APP"
    echo "Trying alternative location..."
    FASTAPI_APP=$(find ~/mlflow-env -name "fastapi_app.py" -type f 2>/dev/null | head -1)
    if [ -z "$FASTAPI_APP" ]; then
        echo "Error: Could not find fastapi_app.py"
        exit 1
    fi
    echo "Found at: $FASTAPI_APP"
fi

# Backup original file
echo "Creating backup..."
sudo cp "$FASTAPI_APP" "${FASTAPI_APP}.backup.$(date +%Y%m%d_%H%M%S)"

# Patch the file to disable host validation
echo "Patching MLflow to disable host validation..."

# Method 1: Comment out the host validation middleware
sudo sed -i 's/from starlette.middleware.trustedhost import TrustedHostMiddleware/# from starlette.middleware.trustedhost import TrustedHostMiddleware/g' "$FASTAPI_APP" 2>/dev/null || true

# Method 2: Remove or comment out TrustedHostMiddleware usage
sudo sed -i 's/app.add_middleware(TrustedHostMiddleware/# app.add_middleware(TrustedHostMiddleware/g' "$FASTAPI_APP" 2>/dev/null || true

# Method 3: If the above don't work, we'll add a patch to allow all hosts
if grep -q "TrustedHostMiddleware" "$FASTAPI_APP"; then
    echo "Found TrustedHostMiddleware, attempting to disable it..."
    # Try to replace with a no-op or allow all hosts
    sudo python3 << EOF
import re

with open("$FASTAPI_APP", "r") as f:
    content = f.read()

# Replace TrustedHostMiddleware with a version that allows all hosts
content = re.sub(
    r'app\.add_middleware\(TrustedHostMiddleware[^)]+\)',
    '# app.add_middleware(TrustedHostMiddleware(...))  # Disabled to allow IP access',
    content
)

# Also try to find and comment out the import
content = re.sub(
    r'^from starlette\.middleware\.trustedhost import TrustedHostMiddleware',
    '# from starlette.middleware.trustedhost import TrustedHostMiddleware  # Disabled',
    content,
    flags=re.MULTILINE
)

with open("$FASTAPI_APP", "w") as f:
    f.write(content)
EOF
fi

echo "Patch applied. Restarting MLflow service..."
sudo systemctl restart mlflow

echo "Waiting for service to start..."
sleep 5

# Check status
sudo systemctl status mlflow --no-pager | head -20

echo ""
echo "Done! MLflow should now accept requests with IP addresses."
echo "Test from your local machine:"
echo "  curl http://YOUR-EC2-IP:5000"
echo "  python3 -c \"import mlflow; mlflow.set_tracking_uri('http://YOUR-EC2-IP:5000'); print('Connected!')\""



