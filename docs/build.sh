#!/bin/bash

# Build script for Sphinx documentation

set -e

echo "Building Agentic RAG Documentation..."

# Check if we're in the docs directory
if [ ! -f "conf.py" ]; then
    echo "Error: This script must be run from the docs directory"
    exit 1
fi

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf _build/

# Check if Sphinx is installed
if ! command -v sphinx-build &> /dev/null; then
    echo "Error: sphinx-build not found. Please install Sphinx:"
    echo "pip install sphinx sphinx-rtd-theme"
    exit 1
fi

# Build HTML documentation
echo "Building HTML documentation..."
sphinx-build -b html . _build/html

# Build PDF documentation (if LaTeX is available)
if command -v pdflatex &> /dev/null; then
    echo "Building PDF documentation..."
    sphinx-build -b latex . _build/latex
    cd _build/latex
    make
    cd ../..
else
    echo "LaTeX not found, skipping PDF build"
fi

echo "Documentation build complete!"
echo "HTML documentation available at: _build/html/index.html"

# Optional: Open documentation in browser
if command -v xdg-open &> /dev/null; then
    xdg-open _build/html/index.html
elif command -v open &> /dev/null; then
    open _build/html/index.html
elif command -v start &> /dev/null; then
    start _build/html/index.html
fi