#!/usr/bin/env bash
# Quick Start Script for AlgoBrain Web App

echo "🚀 AlgoBrain Web App - Quick Start"
echo "===================================="

# Check Python version
python_version=$(python --version 2>&1)
echo "✓ Python: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "📥 Installing dependencies..."
pip install -r requirements.txt

# Run the app
echo "🎯 Launching Gradio app..."
echo ""
echo "================================"
echo "✅ App is running at:"
echo "   Local: http://localhost:7860"
echo "================================"
echo ""

python app.py
