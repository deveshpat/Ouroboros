# Ouroboros Codespace Setup Guide

This directory contains the GitHub Codespaces configuration for persistent development environment setup.

## Files

- **devcontainer.json** - Main Codespace configuration
- **post-create.sh** - Runs once after container creation (installs Ollama, Pi Code, dependencies)
- **post-start.sh** - Runs every time the Codespace starts (ensures services are running)

## What Gets Set Up Automatically

### On First Codespace Creation (`post-create.sh`)
✅ Ollama installation (if not present)
✅ Node.js via NVM (if not present)
✅ Pi Code global installation (if not present)
✅ Python dependencies from requirements.txt (if present)
✅ Ollama service starts in background

### On Every Codespace Start (`post-start.sh`)
✅ Ensures Ollama service is running

## Forwarded Ports

- **11434** - Ollama API (auto-forward disabled to avoid noise)
- **3000** - Development server
- **8000, 8080** - Additional services

## Customizations

The setup includes VSCode extensions for:
- Python development (Pylance, Ruff)
- Formatting (Prettier)
- Git & GitHub integration

## Manual Commands

If you need to manually restart services:

```bash
# Start Ollama
ollama serve

# Check Ollama status
curl http://localhost:11434/api/tags

# Start Pi Code
pi

# View Ollama logs
tail -f /tmp/ollama.log
```

## Troubleshooting

### Ollama not starting?
```bash
# Start manually
ollama serve > /tmp/ollama.log 2>&1 &
```

### Pi Code not found?
```bash
# Reinstall
npm install -g @mariozechner/pi-coding-agent

# Or use via npm
npx @mariozechner/pi-coding-agent
```

### Check what's running
```bash
ps aux | grep -E 'ollama|node'
```

## Adding to Git

These files should be committed to preserve the setup across all Codespaces:

```bash
git add .devcontainer/
git commit -m "Add persistent Codespace setup for Ollama and Pi Code"
git push
```

Any new Codespace created from this repo will automatically run through the setup!
