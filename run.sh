#!/bin/bash

# Quick run script - assumes setup.sh has been run

if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Please run ./setup.sh first"
    exit 1
fi

echo "Starting PDF Translation POC..."
source venv/bin/activate
python app.py