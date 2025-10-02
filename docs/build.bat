@echo off
REM Build script for Sphinx documentation on Windows

echo Building Agentic RAG Documentation...

REM Check if we're in the docs directory
if not exist "conf.py" (
    echo Error: This script must be run from the docs directory
    exit /b 1
)

REM Clean previous builds
echo Cleaning previous builds...
if exist "_build" rmdir /s /q "_build"

REM Check if Sphinx is installed
sphinx-build --version >nul 2>&1
if errorlevel 1 (
    echo Error: sphinx-build not found. Please install Sphinx:
    echo pip install sphinx sphinx-rtd-theme
    exit /b 1
)

REM Build HTML documentation
echo Building HTML documentation...
sphinx-build -b html . _build/html

if errorlevel 1 (
    echo Error occurred during HTML build
    exit /b 1
)

echo Documentation build complete!
echo HTML documentation available at: _build/html/index.html

REM Optional: Open documentation in browser
start _build/html/index.html