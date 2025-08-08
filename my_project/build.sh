#!/usr/bin/env bash
# exit on error
set -o errexit

# Install system packages required for compilation
apt-get update
apt-get install -y build-essential libssl-dev libffi-dev

# Install Python dependencies
pip install -r requirements.txt

# Run Django commands
python manage.py collectstatic --no-input
python manage.py migrate
