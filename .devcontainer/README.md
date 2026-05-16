# Codespace

Purpose -> persistent dev env.

## Files

`devcontainer.json` -> config.
`post-create.sh` -> first boot setup.
`post-start.sh` -> service restart.

## Setup

first boot -> install Ollama + Node/NVM + Pi Code + Python deps.
every boot -> ensure Ollama running.

## Ports

`11434` -> Ollama.
`3000` -> dev server.
`8000`, `8080` -> extra services.

## Commands

```bash
ollama serve
curl http://localhost:11434/api/tags
pi
ps aux | grep -E 'ollama|node'
```
