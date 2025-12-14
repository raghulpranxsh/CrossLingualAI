#!/bin/bash

# SecureNews - Run Script
# This script sets up and runs the SecureNews application

echo "🚀 Starting SecureNews Application..."
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3.8 or higher."
    exit 1
fi

# Check if ffmpeg is installed
if ! command -v ffmpeg &> /dev/null; then
    echo "⚠️  FFmpeg is not installed. Installing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install ffmpeg
        else
            echo "❌ Please install Homebrew first: https://brew.sh"
            echo "   Then run: brew install ffmpeg"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        sudo apt-get update && sudo apt-get install -y ffmpeg
    else
        echo "❌ Please install FFmpeg manually for your OS"
        exit 1
    fi
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install/upgrade pip
echo "📥 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📦 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads outputs

echo ""
echo "✅ Setup complete!"
echo ""
echo "🌐 Starting Flask server..."
echo "   Open your browser and navigate to: http://localhost:5001"
echo ""
echo "⚠️  Note: First run will download ML models (Whisper, NLLB, etc.)"
echo "   This may take 10-15 minutes depending on your internet connection."
echo ""

# Run the application
python3 app.py

