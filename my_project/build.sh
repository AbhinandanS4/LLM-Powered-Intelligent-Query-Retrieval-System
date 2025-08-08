#!/usr/bin/env bash
# exit on error
set -o errexit

# Install required build tools for C++ extensions
apt-get update
apt-get install -y build-essential

# Install Python dependencies
pip install -r requirements.txt

# Run Django commands
python manage.py collectstatic --no-input
python manage.py migrate
