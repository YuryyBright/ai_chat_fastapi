# scripts/setup_ollama.sh
#!/bin/bash
# Setup script for pulling common Ollama models

echo "==================================="
echo "Ollama Models Setup"
echo "==================================="

MODELS=(
    "llama2"
    "mistral"
    "codellama"
    "llama2:13b"
    "mistral:7b"
)

echo "This script will pull the following models:"
for model in "${MODELS[@]}"; do
    echo "  - $model"
done

read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 1
fi

for model in "${MODELS[@]}"; do
    echo ""
    echo "Pulling $model..."
    docker exec -it ollama ollama pull "$model"
    
    if [ $? -eq 0 ]; then
        echo "✓ $model pulled successfully"
    else
        echo "✗ Failed to pull $model"
    fi
done

echo ""
echo "==================================="
echo "Setup complete!"
echo "==================================="
echo ""
echo "Available models:"
docker exec -it ollama ollama list