# Render Build Script
#!/usr/bin/env bash
# Exit on error
set -o errexit

# Install backend dependencies
cd backend
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn

echo "Build completed successfully!"
