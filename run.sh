#!/bin/bash

# Run script that keeps venv activated after exit

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

echo "=========================================="
echo "Starting PDF Translation POC..."
echo "=========================================="
echo ""

# Check if already in venv
if [[ "$VIRTUAL_ENV" != "" ]]; then
    echo "✓ Virtual environment already active"
else
    echo "Activating virtual environment..."
    source venv/bin/activate
    echo "✓ Virtual environment activated"
fi

echo "✓ Starting Flask application..."
echo ""
echo "Open your browser to: http://127.0.0.1:5000"
echo "Press Ctrl+C to stop the server"
echo ""

python app.py

# Note: Virtual environment will remain active after script exits
# Run 'deactivate' to exit the virtual environment