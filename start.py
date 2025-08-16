#!/usr/bin/env python3
"""
Simple startup script for EAMCET College Predictor
"""

import sys
import os
import webbrowser
import time
from threading import Timer

def open_browser():
    """Open browser after a short delay"""
    webbrowser.open('http://127.0.0.1:5001')

if __name__ == "__main__":
    print("🎓 Starting EAMCET College Predictor...")
    print("📍 Server will run on: http://127.0.0.1:5001")
    print("🌐 External access on: http://0.0.0.0:5001")
    print("⏳ Loading AI models and initializing server...")
    print("-" * 50)
    
    # Schedule browser opening after 3 seconds
    Timer(3.0, open_browser).start()
    
    # Import and run the app
    from app import app
    
    try:
        app.run(debug=True, port=5001, host='0.0.0.0')
    except KeyboardInterrupt:
        print("\n👋 Shutting down EAMCET College Predictor. Goodbye!")
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        sys.exit(1)
