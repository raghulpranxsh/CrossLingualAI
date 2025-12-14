@echo off
REM SecureNews - Run Script for Windows
REM This script sets up and runs the SecureNews application

echo 🚀 Starting SecureNews Application...
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

REM Check if ffmpeg is installed
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo ⚠️  FFmpeg is not installed.
    echo    Please download and install FFmpeg from: https://ffmpeg.org/download.html
    echo    Make sure to add it to your PATH.
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade pip
echo 📥 Upgrading pip...
python -m pip install --upgrade pip

REM Install requirements
echo 📦 Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt

REM Create necessary directories
if not exist "uploads" mkdir uploads
if not exist "outputs" mkdir outputs

echo.
echo ✅ Setup complete!
echo.
echo 🌐 Starting Flask server...
echo    Open your browser and navigate to: http://localhost:5001
echo.
echo ⚠️  Note: First run will download ML models (Whisper, NLLB, etc.)
echo    This may take 10-15 minutes depending on your internet connection.
echo.

REM Run the application
python app.py

pause

