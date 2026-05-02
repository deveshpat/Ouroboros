#!/bin/bash
set -e

echo "🚀 Starting Ouroboros development environment setup..."

# Update package managers
echo "📦 Updating package managers..."
sudo apt-get update -qq

# Install Ollama if not already installed
if ! command -v ollama &> /dev/null; then
    echo "📥 Installing Ollama..."
    curl -fsSL https://ollama.ai/install.sh | sh
else
    echo "✅ Ollama already installed"
fi

# Install Node.js and npm via NVM if not already present
if ! command -v node &> /dev/null; then
    echo "📥 Installing Node.js via nvm..."
    curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
    export NVM_DIR="$HOME/.nvm"
    [ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
    nvm install node
    nvm use node
else
    echo "✅ Node.js already installed"
fi

# Install Pi Code globally if not already installed
if ! command -v pi &> /dev/null; then
    echo "📥 Installing Pi Code..."
    npm install -g @mariozechner/pi-coding-agent
else
    echo "✅ Pi Code already installed"
fi

# Ensure Python dependencies are installed
if [ -f "requirements.txt" ]; then
    echo "📥 Installing Python dependencies..."
    pip install -q -r requirements.txt
fi

# Start Ollama in the background
echo "🎯 Starting Ollama service..."
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 2
    echo "✅ Ollama started (PID: $!)"
else
    echo "✅ Ollama already running"
fi

echo ""
echo "✨ Setup complete!"
echo ""
echo "Available commands:"
echo "  - ollama serve          (Ollama API server)"
echo "  - pi                    (Pi Code agent)"
echo "  - python <script.py>    (Run Python scripts)"
echo ""
echo "Forwarded ports:"
echo "  - Ollama API: localhost:11434"
echo ""
