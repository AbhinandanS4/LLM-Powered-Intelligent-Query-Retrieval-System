#!/usr/bin/env bash
# exit on error
set -o errexit

# Install necessary system packages for compilation
apt-get update
apt-get install -y build-essential libssl-dev libffi-dev pkg-config python3-dev

# Install Python dependencies from requirements.txt
pip install -r requirements.txt

# Run Django commands to prepare the application
python manage.py collectstatic --no-input
python manage.py migrate

# Finally, start the Gunicorn server
# Make sure the path to your wsgi file is correct here.
exec gunicorn my_project.my_project.wsgi:application
