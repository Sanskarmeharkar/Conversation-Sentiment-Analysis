"""
Conversation Sentiment & Emotion Analysis Dashboard Entry Point
Redirects or executes main application from app/app.py
"""

from pathlib import Path
import sys

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Execute main application
import app
