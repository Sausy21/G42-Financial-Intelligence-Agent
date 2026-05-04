@echo off
REM G42 Financial Intelligence Agent — Windows Setup
REM Creates a virtual environment and installs all dependencies.
REM
REM Usage: setup.bat

echo ===================================================
echo   G42 Financial Intelligence Agent — Setup
echo ===================================================
echo.

REM Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

REM Create virtual environment
echo -^> Creating virtual environment...
python -m venv finAgent

REM Activate
echo -^> Activating virtual environment...
call finAgent\Scripts\activate.bat

REM Upgrade pip
echo -^> Upgrading pip...
pip install --upgrade pip --quiet

REM Install dependencies
echo -^> Installing dependencies (this may take a few minutes)...
pip install -r requirements.txt --quiet

REM Copy .env
if not exist .env (
    echo -^> Creating .env from .env.example...
    copy .env.example .env >nul
    echo.
    echo   IMPORTANT: Edit .env and add your API keys
    echo.
)

REM Generate sample data
echo -^> Generating sample financial data...
python data\generate_sample.py

REM Run tests
echo.
echo -^> Running test suite...
python -m pytest tests/ -v --tb=short

echo.
echo ===================================================
echo   Setup complete!
echo ===================================================
echo.
echo   To activate:  finAgent\Scripts\activate
echo   To run UI:    streamlit run ui\app.py
echo   To deactivate: deactivate
echo.
pause
