@echo off
REM Quick Start Script for AlgoBrain Web App (Windows)

echo.
echo 🚀 AlgoBrain Web App - Quick Start
echo ====================================

REM Check Python
python --version
if %errorlevel% neq 0 (
    echo ❌ Python not found. Please install Python 3.8+
    exit /b 1
)

REM Create virtual environment if it doesn't exist
if not exist "venv" (
    echo 📦 Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo 🔌 Activating virtual environment...
call venv\Scripts\activate.bat

REM Install requirements
echo 📥 Installing dependencies...
pip install -r requirements.txt

REM Run the app
echo 🎯 Launching Gradio app...
echo.
echo ================================
echo ✅ App will open at:
echo    Local: http://localhost:7860
echo ================================
echo.

python app.py

pause
