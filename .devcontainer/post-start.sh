#!/bin/bash
set -e

echo "🚀 Starting Ouroboros environment..."

# Ensure Ollama service is running
if ! pgrep -x "ollama" > /dev/null; then
    echo "🎯 Starting Ollama service..."
    ollama serve > /tmp/ollama.log 2>&1 &
    sleep 2
    echo "✅ Ollama started (PID: $!)"
else
    echo "✅ Ollama already running"
fi

echo "✨ Environment ready!"
