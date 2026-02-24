@echo off
setlocal enabledelayedexpansion

echo ==================================================
echo   Lidar Processing Toolkit - One-Click Setup
echo ==================================================

:: 1. Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    echo Please install Python 3.12 from https://www.python.org/
    echo and make sure to check "Add Python to PATH" during installation.
    pause
    exit /b
)

:: 2. Create Virtual Environment
if not exist "venv" (
    echo [1/4] Creating virtual environment...
    python -m venv venv
) else (
    echo [1/4] Virtual environment already exists.
)

:: 3. Install/Update Dependencies
echo [2/4] Installing/Updating dependencies...
:: Explicitly use the python inside venv to install
venv\Scripts\python.exe -m pip install --upgrade pip
:: Install setuptools first to fix pkg_resources issue for pyarrow
venv\Scripts\python.exe -m pip install setuptools
venv\Scripts\pip.exe install -r requirements.txt

:: 4. Ensure Directory Structure
echo [3/4] Ensuring data/ and outputs/ directories exist...
if not exist "data" mkdir data
if not exist "outputs" mkdir outputs

:: 5. Launch Application
echo [4/4] Starting Streamlit application...
echo --------------------------------------------------
echo Tips: Place your .pcd files in the "data" folder.
echo --------------------------------------------------
:: Use the absolute path to the venv python to avoid using system python
:: The --global.developmentMode=false and browser.gatherUsageStats=false help skip some prompts
venv\Scripts\python.exe -m streamlit run Home.py --browser.gatherUsageStats=false

pause
