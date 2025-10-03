# WSGI configuration for PythonAnywhere
# This file tells PythonAnywhere how to run your FastAPI application

import sys
import os

# Add your project directory to the sys.path
project_home = '/home/YOUR_USERNAME/bdc-2'  # UPDATE THIS with your PythonAnywhere username
if project_home not in sys.path:
    sys.path.insert(0, project_home)

# Load environment variables from .env file
from dotenv import load_dotenv
env_path = os.path.join(project_home, '.env')
load_dotenv(env_path)

# Import your FastAPI app
from app import app as application

# PythonAnywhere expects the WSGI application to be named "application"
# FastAPI app is already ASGI-compatible, we need to wrap it
