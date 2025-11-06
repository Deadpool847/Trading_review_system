#!/bin/bash
# Daily Review Machine Startup Script

echo "🚀 Starting Daily Review Machine..."
echo ""

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/bars data/trades data/cache models reports logs
echo "✓ Directories created"

# Check dependencies
echo "📦 Checking dependencies..."
if python -c "import streamlit, polars, plotly, sklearn, lightgbm, growwapi" 2>/dev/null; then
    echo "✓ All dependencies installed"
else
    echo "❌ Missing dependencies. Installing..."
    pip install -r backend/requirements.txt
fi

echo ""
echo "════════════════════════════════════════════════════════"
echo "  📊 Daily Review Machine Ready!"
echo "════════════════════════════════════════════════════════"
echo ""
echo "Choose your interface:"
echo ""
echo "  1️⃣  Streamlit App (Interactive UI)"
echo "      streamlit run streamlit_app.py"
echo ""
echo "  2️⃣  CLI (Automation)"
echo "      python cli.py run --date 2025-01-15 --scope daily"
echo ""
echo "════════════════════════════════════════════════════════"
echo ""

# Ask user preference
read -p "Launch Streamlit app now? (y/n): " choice

if [ "$choice" = "y" ] || [ "$choice" = "Y" ]; then
    echo "🌐 Starting Streamlit..."
    streamlit run streamlit_app.py
else
    echo "👋 Run 'streamlit run streamlit_app.py' when ready!"
fi
