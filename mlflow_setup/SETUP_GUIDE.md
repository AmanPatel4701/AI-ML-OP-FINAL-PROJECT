# Complete Remote MLflow Setup Guide

This is a detailed step-by-step guide to set up your remote MLflow infrastructure (EC2 + Neon PostgreSQL + S3) according to the project requirements.

## Overview

You need to set up:

1. **Neon PostgreSQL** - Database backend for MLflow
2. **AWS S3 Bucket** - Artifact storage
3. **AWS EC2 Instance** - MLflow server host
4. **Configure MLflow** - Connect everything together

---

## Part 1: Set Up Neon PostgreSQL Database

### Step 1.1: Create Neon Account

1. Go to https://neon.com
2. Click "Sign Up" (or "Log In" if you have an account)
3. You can use GitHub, Google, or email to sign up
4. **Free tier is sufficient** for this project

### Step 1.2: Create a Project

1. After logging in, click "Create a project"
2. Choose a project name (e.g., "mlflow-backend")
3. Select a region close to your EC2 region (e.g., `us-east-1`)
4. Click "Create project"

![Alt text for the image](../images/create_neon_project.png "Optional title text")

### Step 1.3: Get Connection String

1. In your Neon project dashboard, you'll see connection details
2. Look for "Connection string" section
3. Copy the connection string - it looks like:
   ```
   psql 'postgresql://neondb_owner:npg_P7Wm1xFHEZuo@ep-late-rice-adl7h7wz-pooler.c-2.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require'
   ```

![Alt text for the image](../images/connection_string.png "Optional title text")

4. **IMPORTANT**: Save this connection string securely - you'll need it later
5. Note: The default database is usually `neondb` or `defaultdb`. You can create a new database called `mlflowdb` if you prefer, or use the default one.

### Step 1.4: Create MLflow Database (Optional but Recommended)

1. In Neon dashboard, go to "SQL Editor"
2. Run this SQL to create a dedicated database:
   ```sql
   CREATE DATABASE mlflowdb;
   ```

![Alt text for the image](../images/create_db.png "Optional title text")

3. Update your connection string to use `mlflowdb` instead of `neondb`:
   ```
   postgresql://neondb_owner:npg_P7Wm1xFHEZuo@ep-late-rice-adl7h7wz-pooler.c-2.us-east-1.aws.neon.tech/mlflowdb?sslmode=require&channel_binding=require
   ```

![Alt text for the image](../images/change_db_to_mlflowdb.png "Optional title text")

### Step 1.5: Configure Network Access

1. In Neon dashboard, go to "Settings" → "IP Allowlist"
2. You'll need to allow your EC2 instance IP (we'll do this after creating EC2)
3. For now, you can temporarily allow all IPs (0.0.0.0/0) for testing, but restrict it later

**Checkpoint**: You should have:

- Neon account created
- Connection string saved
- Database ready (mlflowdb or default)

---

## Part 2: Set Up AWS S3 Bucket

### Step 2.1: Log into AWS Console

1. Go to https://console.aws.amazon.com
2. Log in with your AWS account
3. If you don't have an account, create one (free tier available)

### Step 2.2: Create S3 Bucket

1. In AWS Console, search for "S3" or go to Services → S3
2. Click "Create bucket"
3. Configure the bucket:

   - **Bucket name**: `mlflow-artifacts-yourname` (must be globally unique)

     ![1765181968241](../images/1765181968241.png)
   - **Region**: Choose same region as your EC2 (e.g., `us-east-1`)
   - **Object Ownership**: ACLs disabled (recommended)
   - **Block Public Access**: Keep all settings enabled (for security)
   - **Bucket Versioning**: Optional (can enable for better tracking)
   - **Default encryption**: Enable (SSE-S3 is fine)
4. Click "Create bucket"

   ![1765182447164](../images/1765182447164.png)

### Step 2.3: Create IAM User for S3 Access

1. In AWS Console, search for "IAM" or go to Services → IAM
2. Click "Users" in the left sidebar
3. Click "Create user"
4. Enter username: `mlflow-s3-user`
5. Click "Next"

   ![1765182678549](../images/1765182678549.png)

### Step 2.4: Attach S3 Policy

1. Select "Attach policies directly"
2. Search for and select: **AmazonS3FullAccess** (or create a custom policy with only your bucket)

   ![1765183900490](../images/1765183900490.png)
3. Click "Next" → "Create user"

   ![1765183930120](../images/1765183930120.png)

   ![1765183956212](../images/1765183956212.png)

### Step 2.5: Create Access Keys

1. Click on the user you just created (`mlflow-s3-user`)
2. Go to "Security credentials" tab
3. Scroll to "Access keys" section
4. Click "Create access key"
5. Select "Application running outside AWS"

   ![1765184041413](../images/1765184041413.png)
6. Click "Next" → "Create access key"

   ![1765184062827](../images/1765184062827.png)
7. **IMPORTANT**: Copy both:

   - **Access Key ID** (e.g., `AKIAIOSFODNN7EXAMPLE`)
   - **Secret Access Key** (e.g., `wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY`)
   - **Save these securely** - you won't see the secret key again!

     ![1765184095073](../images/1765184095073.png)

**Checkpoint**: You should have:

- S3 bucket created
- IAM user with access keys
- Access Key ID and Secret Access Key saved

---

## Part 3: Set Up AWS EC2 Instance

### Step 3.1: Launch EC2 Instance

1. In AWS Console, search for "EC2" or go to Services → EC2
2. Click "Launch instance" (orange button)

   ![1765184318129](../images/1765184318129.png)
3. Configure instance:

   - **Name**: `mlflow-server`
   - **AMI**: Ubuntu Server 22.04 LTS (free tier eligible)
   - **Instance type**: `t3.micro` (free tier) or `t3.small` (better performance)
   - **Key pair**:
     - If you have one, select it
     - If not, click "Create new key pair"
     - Name: `mlflow-key`
     - Key pair type: RSA
     - File format: `.pem`
     - Click "Create key pair" - **Download the .pem file and save it securely!**

       ![1765184617375](../images/1765184617375.png)
   - **Network settings**:
     - Click "Edit"
     - Security group: Create new security group
     - Name: `mlflow-sg`
     - Add rules:

       - **SSH** (port 22): Source = My IP
       - **Custom TCP** (port 5000): Source = My IP (or 0.0.0.0/0 for testing)
     - Click "Save"

       ![1765185213546](../images/1765185213546.png)
   - **Configure storage**: 8 GB (free tier) or 20 GB (recommended)
4. Click "Launch instance"

   ![1765189219412](../images/1765189219412.png)

### Step 3.2: Get EC2 Public IP

1. Wait for instance to be "Running" (green checkmark)

  ![1765190781866](../images/1765190781866.png)
2. Click on the instance ID
3. In the details panel, find "Public IPv4 address"
4. **Copy this IP** - you'll need it (e.g., `54.123.45.67`)

  ![1765190620776](../images/1765190620776.png)

### Step 3.3: Update Neon IP Allowlist

1. Go back to Neon dashboard
2. Settings → IP Allowlist
3. Add your EC2 public IP (or allow all for testing: `0.0.0.0/0`)

### Step 3.4: SSH into EC2 Instance

1. Open terminal on your local machine
2. Navigate to where you saved the `.pem` file
3. Set correct permissions:

   ```bash
   chmod 400 mlflow-key.pem
   ```
4. SSH into instance:

   ```bash
   ssh -i mlflow-key.pem ubuntu@YOUR-EC2-PUBLIC-IP
   ```

   Replace `YOUR-EC2-PUBLIC-IP` with your actual IP
5. If prompted, type `yes` to accept the fingerprint

**Checkpoint**: You should be logged into your EC2 instance

---

## Part 4: Install and Configure MLflow on EC2

### Step 4.1: Update System (on EC2)

```bash
sudo apt-get update
sudo apt-get upgrade -y
```

### Step 4.2: Install Python and Dependencies

```bash
# Install Python and pip
sudo apt-get install -y python3-pip python3-venv

# Install PostgreSQL client (for testing)
sudo apt-get install -y postgresql-client

# Install MLflow and required packages
pip3 install mlflow psycopg2-binary boto3
```

### Step 4.3: Test PostgreSQL Connection

```bash
# Test connection (replace with your Neon connection string)
psql "postgresql://username:password@ep-xxxx-xxxx.us-east-1.aws.neon.tech/mlflowdb?sslmode=require"
```

If it connects successfully, type `\q` to quit.

### Step 4.4: Create Environment File

```bash
# Create environment file with your credentials
nano ~/.mlflow_env
```

Paste the following (replace with YOUR actual values):

```bash
export MLFLOW_BACKEND_STORE_URI="postgresql://username:password@ep-xxxx-xxxx.us-east-1.aws.neon.tech/mlflowdb?sslmode=require"
export MLFLOW_DEFAULT_ARTIFACT_ROOT="s3://mlflow-artifacts-yourname/mlflow"
export AWS_ACCESS_KEY_ID="YOUR-ACCESS-KEY-ID"
export AWS_SECRET_ACCESS_KEY="YOUR-SECRET-ACCESS-KEY"
export AWS_DEFAULT_REGION="us-east-1"
```

**Replace:**

- `username:password@ep-xxxx-xxxx...` with your Neon connection string
- `mlflow-artifacts-yourname` with your actual S3 bucket name
- `YOUR-ACCESS-KEY-ID` with your IAM access key
- `YOUR-SECRET-ACCESS-KEY` with your IAM secret key
- `us-east-1` with your actual region

Save and exit: `Ctrl+X`, then `Y`, then `Enter`

### Step 4.5: Test MLflow Server (Manual Start)

```bash
# Load environment variables
source ~/.mlflow_env

# Start MLflow server
mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port 5000
```

You should see:

```
[INFO] Starting gunicorn 20.1.0
[INFO] Listening at: http://0.0.0.0:5000
```

**Test it**: Open browser and go to `http://YOUR-EC2-PUBLIC-IP:5000`

If it works, press `Ctrl+C` to stop it.

### Step 4.6: Set Up MLflow as System Service (Recommended)

```bash
# Create systemd service file
sudo nano /etc/systemd/system/mlflow.service
```

Paste this content:

```ini
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=ubuntu
EnvironmentFile=/home/ubuntu/.mlflow_env
ExecStart=/usr/local/bin/mlflow server \
    --backend-store-uri "$MLFLOW_BACKEND_STORE_URI" \
    --default-artifact-root "$MLFLOW_DEFAULT_ARTIFACT_ROOT" \
    --host 0.0.0.0 \
    --port 5000
Restart=always
RestartSec=10

[Install]
WantedBy=multiuser.target
```

Save and exit: `Ctrl+X`, `Y`, `Enter`

### Step 4.7: Start and Enable MLflow Service

```bash
# Reload systemd
sudo systemctl daemon-reload

# Enable service (start on boot)
sudo systemctl enable mlflow

# Start service
sudo systemctl start mlflow

# Check status
sudo systemctl status mlflow
```

You should see "active (running)" in green.

### Step 4.8: Verify MLflow is Running

1. Check logs:

   ```bash
   sudo journalctl -u mlflow -f
   ```

   Press `Ctrl+C` to exit
2. Test in browser: `http://YOUR-EC2-PUBLIC-IP:5000`

   - You should see the MLflow UI
   - Try creating a test experiment

**Checkpoint**: MLflow server should be running and accessible

---

## Part 5: Configure Local Machine

### Step 5.1: Set MLflow Tracking URI

On your local machine (in the project directory):

```bash
# Option 1: Set environment variable
export MLFLOW_TRACKING_URI="http://YOUR-EC2-PUBLIC-IP:5000"

# Option 2: Add to your shell profile (permanent)
echo 'export MLFLOW_TRACKING_URI="http://YOUR-EC2-PUBLIC-IP:5000"' >> ~/.zshrc
source ~/.zshrc
```

### Step 5.2: Test Connection

```bash
# Test if you can reach MLflow
curl http://YOUR-EC2-PUBLIC-IP:5000

# Or in Python
python3 -c "import mlflow; mlflow.set_tracking_uri('http://YOUR-EC2-PUBLIC-IP:5000'); print('Connected!')"
```

---

## Part 6: Run Your Training Scripts

Now you can run your training scripts with the remote MLflow:

```bash
# Make sure you're in the project directory and virtual environment is activated
cd household-electricity-forecasting-mlops
source .venv/bin/activate

# Run training scripts
python scripts/train_xgboost.py --tracking-uri http://YOUR-EC2-PUBLIC-IP:5000
python scripts/train_gbm.py --tracking-uri http://YOUR-EC2-PUBLIC-IP:5000
python scripts/train_glm.py --tracking-uri http://YOUR-EC2-PUBLIC-IP:5000
```

---

## Troubleshooting

### Can't connect to MLflow UI

- Check EC2 security group allows port 5000
- Verify MLflow service is running: `sudo systemctl status mlflow`
- Check EC2 instance is running

### Database connection errors

- Verify Neon connection string is correct
- Check Neon IP allowlist includes EC2 IP
- Test connection: `psql "YOUR-CONNECTION-STRING"`

### S3 access denied

- Verify IAM credentials are correct
- Check IAM user has S3 permissions
- Verify bucket name is correct
- Check AWS region matches

### View MLflow logs

```bash
# On EC2
sudo journalctl -u mlflow -f
```

---

## Security Best Practices

1. **Never commit credentials to Git**
2. Restrict EC2 security group to your IP only
3. Use IAM roles instead of access keys when possible
4. Rotate credentials regularly
5. Consider using AWS Secrets Manager for credentials

---

## Cost Estimation (Free Tier)

- **Neon PostgreSQL**: Free tier includes 0.5 GB storage
- **AWS EC2**: Free tier includes 750 hours/month of t2.micro
- **AWS S3**: Free tier includes 5 GB storage
- **Total**: Should be $0/month if within free tier limits

---

## Next Steps

After setup:

1. Run data cleaning: `python scripts/01_clean_and_split.py`
2. Run H2O AutoML: `python scripts/02_h2o_automl.py`
3. Train models with remote MLflow
4. View results in MLflow UI
5. Register models in Model Registry
6. Deploy FastAPI application

---

## Quick Reference

**Neon Connection String:**

```
postgresql://user:pass@ep-xxxx-xxxx.region.aws.neon.tech/mlflowdb?sslmode=require
```

**S3 Bucket:**

```
s3://mlflow-artifacts-yourname/mlflow
```

**MLflow Tracking URI:**

```
http://YOUR-EC2-PUBLIC-IP:5000
```

**EC2 SSH:**

```bash
ssh -i mlflow-key.pem ubuntu@YOUR-EC2-PUBLIC-IP
```
