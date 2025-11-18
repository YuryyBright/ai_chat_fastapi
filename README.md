# LLM Service - Unified Multi-Provider API

A production-ready FastAPI service for working with multiple Large Language Model providers including Ollama (local models), HuggingFace, and OpenAI GPT models. Features streaming generation, model fine-tuning, and Docker support.

## 🚀 Features

- **Multiple LLM Providers**
  - Ollama for local models (Llama2, Mistral, etc.)
  - HuggingFace models (local and remote)
  - OpenAI GPT models

- **Text Generation**
  - Synchronous generation
  - Streaming generation with Server-Sent Events
  - Customizable parameters (temperature, top_p, max_tokens, etc.)

- **Model Fine-Tuning**
  - Fine-tune HuggingFace models on custom datasets
  - Background training with progress tracking
  - Save and deploy fine-tuned models

- **Production Ready**
  - Docker and Docker Compose support
  - Comprehensive error handling
  - Request validation with Pydantic
  - Structured logging
  - Health checks and monitoring

- **Clean Architecture**
  - Modular provider system
  - Easy to extend with new providers
  - Single configuration file
  - Type safety with Pydantic

## 📁 Project Structure

# 📁 Project Structure

Complete directory structure of Local LLM Service.

```
llm-service/
│
├── 📄 README.md                    # Project overview
├── 📄 QUICKSTART.md                # Quick start guide
├── 📄 INSTALLATION.md              # Detailed installation
├── 📄 EXAMPLES.md                  # Usage examples
├── 📄 CHANGES_SUMMARY.md           # Summary of changes
├── 📄 LICENSE                      # License file
│
├── ⚙️ Configuration Files
│   ├── .env.example                # Configuration template
│   ├── .env                        # Your config (git-ignored)
│   ├── .gitignore                  # Git ignore rules
│   ├── requirements.txt            # Python dependencies
│   ├── Dockerfile                  # Docker image definition
│   └── docker-compose.yml          # Docker Compose config
│
├── 🐍 CLI Tool
│   └── cli.py                      # Command-line interface
│
├── 📦 Application (app/)
│   │
│   ├── main.py                     # FastAPI application entry
│   ├── config.py                   # Configuration management
│   │
│   ├── 🎯 Core (app/core/)
│   │   │
│   │   ├── providers/              # LLM providers
│   │   │   ├── __init__.py         # Provider manager
│   │   │   ├── base.py             # Base provider interface
│   │   │   └── local_unified_provider.py  # Local models provider
│   │   │
│   │   ├── model_downloader.py     # Model download manager
│   │   └── exceptions.py           # Custom exceptions
│   │
│   ├── 🌐 API Routes (app/api/)
│   │   └── routes/
│   │       ├── generation.py       # Text generation endpoints
│   │       ├── models.py           # Model management endpoints
│   │       └── training.py         # Training endpoints (future)
│   │
│   └── 📋 Schemas (app/schemas/)
│       ├── generation.py           # Generation request/response
│       └── models.py               # Model schemas
│
├── 📂 Data Directories (git-ignored)
│   │
│   ├── models/                     # Local models storage
│   │   ├── .gitkeep
│   │   ├── cache/                  # HuggingFace cache
│   │   └── fine-tuned/             # Fine-tuned models
│   │
│   ├── data/                       # Application data
│   │   └── .gitkeep
│   │
│   └── logs/                       # Application logs
│       ├── .gitkeep
│       └── llm_service.log
│
├── 📚 Documentation (docs/)
│   ├── api/                        # API documentation
│   ├── guides/                     # User guides
│   └── architecture.md             # Architecture overview
│
├── 🧪 Tests (tests/)
│   ├── __init__.py
│   ├── test_providers.py
│   ├── test_api.py
│   └── test_downloader.py
│
└── 📝 Examples (examples/)
    ├── python/
    │   ├── basic_generation.py
    │   ├── streaming.py
    │   └── chatbot.py
    ├── javascript/
    │   └── client.js
    └── curl/
        └── examples.sh
```

---

## 📄 Key Files Description

### Root Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview, features, quick links |
| `QUICKSTART.md` | Get started in 5 minutes |
| `INSTALLATION.md` | Detailed installation instructions |
| `EXAMPLES.md` | Code examples in multiple languages |
| `CHANGES_SUMMARY.md` | Summary of architectural changes |
| `cli.py` | Command-line interface tool |

### Configuration

| File | Purpose |
|------|---------|
| `.env.example` | Configuration template with comments |
| `.env` | Your actual configuration (git-ignored) |
| `requirements.txt` | Python package dependencies |
| `Dockerfile` | Docker image for CPU/GPU |
| `docker-compose.yml` | Docker orchestration |
| `.gitignore` | Files to ignore in git |

### Application Core (`app/`)

| File/Directory | Purpose |
|----------------|---------|
| `main.py` | FastAPI application, startup/shutdown |
| `config.py` | Pydantic settings management |
| `core/providers/` | LLM provider implementations |
| `core/model_downloader.py` | HuggingFace integration |
| `core/exceptions.py` | Custom error handling |
| `api/routes/` | REST API endpoints |
| `schemas/` | Pydantic models for validation |

### Provider System

| File | Purpose |
|------|---------|
| `base.py` | Abstract provider interface |
| `local_unified_provider.py` | Universal local model handler |
| `__init__.py` | Provider manager, orchestration |

### API Endpoints (`app/api/routes/`)

| File | Endpoints |
|------|-----------|
| `generation.py` | `/api/v1/generation/*` |
| `models.py` | `/api/v1/models/*` |
| `training.py` | `/api/v1/training/*` (future) |

---

## 🗂️ Data Directories

### `models/` Structure

```
models/
├── .gitkeep                        # Keep directory in git
├── cache/                          # HuggingFace cache
│   └── models--TheBloke--Llama-2-7B-GGUF/
│       └── snapshots/
│           └── <hash>/
│               └── llama-2-7b.Q4_K_M.gguf
├── llama-2-7b.Q4_K_M.gguf         # GGUF model
├── mistral-7b.Q4_K_M.gguf         # Another GGUF
├── phi-2/                          # HuggingFace model
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer.json
└── fine-tuned/                     # Fine-tuned models
    └── my-model/
```

### `data/` Structure

```
data/
├── .gitkeep
├── conversations/                  # Saved conversations
├── datasets/                       # Training datasets
└── checkpoints/                    # Training checkpoints
```

### `logs/` Structure

```
logs/
├── .gitkeep
├── llm_service.log                # Main application log
├── downloads.log                  # Model downloads
└── errors.log                     # Error logs
```

---

## 🔌 API Endpoints Overview

### Models Management

```
GET    /api/v1/models/list
POST   /api/v1/models/download
POST   /api/v1/models/search
GET    /api/v1/models/files/{repo_id}
GET    /api/v1/models/info/{model}
DELETE /api/v1/models/delete
GET    /api/v1/models/health
```

### Text Generation

```
POST   /api/v1/generation/generate
POST   /api/v1/generation/stream
```

### System

```
GET    /
GET    /health
GET    /info
GET    /docs          # Swagger UI
GET    /redoc         # ReDoc
```

---

## 🧩 Module Dependencies

```
main.py
├── config.py (Settings)
├── core/
│   ├── providers/
│   │   ├── __init__.py (ProviderManager)
│   │   │   └── local_unified_provider.py
│   │   └── base.py
│   ├── model_downloader.py
│   └── exceptions.py
└── api/routes/
    ├── generation.py
    └── models.py

cli.py
└── requests (HTTP client)
    └── API endpoints
```

---

## 📦 Package Layout

```python
# Import examples:

# From application
from app.config import settings
from app.core.providers import ProviderManager
from app.core.model_downloader import ModelDownloader

# From schemas
from app.schemas.generation import GenerationRequest, GenerationResponse
from app.schemas.models import ModelInfo

# From routes
from app.api.routes import generation, models
```

---

## 🔧 Environment Variables Structure

```bash
# Application
APP_NAME=...
VERSION=...
ENVIRONMENT=...
HOST=...
PORT=...
LOG_LEVEL=...

# Directories
MODELS_DIR=...
DATA_DIR=...
CACHE_DIR=...

# Provider
USE_GPU=...
GPU_LAYERS=...
CONTEXT_SIZE=...
BATCH_SIZE=...
N_THREADS=...

# Generation
MAX_TOKENS=...
DEFAULT_TEMPERATURE=...
DEFAULT_TOP_P=...

# Downloads
HUGGINGFACE_TOKEN=...
DOWNLOAD_TIMEOUT=...

# Security
API_KEY_ENABLED=...
API_KEYS=...
```

---

## 📊 File Size Guidelines

| Directory/File | Typical Size |
|----------------|--------------|
| Application code | < 100 KB |
| Virtual environment | ~500 MB |
| GGUF model (Q4_K_M 7B) | ~4 GB |
| GGUF model (Q4_K_M 13B) | ~8 GB |
| HuggingFace model (7B) | ~15 GB |
| Cache (per model) | Variable |
| Logs | < 100 MB |

---

## 🗃️ Git Repository

**Tracked:**
- Source code (`app/`, `cli.py`)
- Documentation (`*.md`)
- Configuration templates (`.env.example`)
- Docker files
- Tests
- Requirements

**Ignored (`.gitignore`):**
- Models (`models/`)
- Data (`data/`)
- Logs (`logs/`)
- Virtual environment (`venv/`)
- Configuration (`.env`)
- Cache files
- Python bytecode (`__pycache__/`)

---

## 🚀 Deployment Structure

### Development
```
llm-service/
├── app/
├── models/
├── .env (development)
└── venv/
```

### Production (Docker)
```
container:
├── /app/
│   ├── app/
│   └── main.py
├── /app/models/ (volume mount)
├── /app/data/ (volume mount)
└── /app/logs/ (volume mount)
```

### Production (Bare Metal)
```
/opt/llm-service/
├── app/
├── venv/
├── models/
├── data/
├── logs/
└── .env (production)
```

---

## 📝 Notes

1. **Models Directory**: Should be on fast SSD for best performance
2. **Logs Directory**: Monitor disk space, implement rotation
3. **Cache Directory**: Can be cleared if disk space needed
4. **Virtual Environment**: Keep separate for each deployment
5. **Configuration**: Never commit `.env` to git

---

**Last Updated:** 2025  
**Version:** 2.0.0

## 🛠️ Installation

### Prerequisites

- Python 3.11+
- Docker and Docker Compose (for containerized deployment)
- NVIDIA GPU with CUDA support (optional, for faster inference)

### Local Development Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd llm-service
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
```

5. **Run the service**
```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Docker Deployment

1. **Configure environment**
```bash
cp .env.example .env
# Edit .env with your configuration
# Set OPENAI_API_KEY if using OpenAI
```

2. **Start services**
```bash
docker-compose up -d
```

3. **Pull Ollama models** (optional)
```bash
docker exec -it ollama ollama pull llama2
docker exec -it ollama ollama pull mistral
```

4. **Check service health**
```bash
curl http://localhost:8000/health
```

## 📖 API Documentation

Once the service is running, access the interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Quick Examples

#### 1. Generate Text (Ollama)

```bash
curl -X POST "http://localhost:8000/api/v1/generation/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms",
    "provider": "ollama",
    "model": "llama2",
    "max_tokens": 500,
    "temperature": 0.7
  }'
```

#### 2. Streaming Generation

```bash
curl -X POST "http://localhost:8000/api/v1/generation/generate/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a short story about a robot",
    "provider": "ollama",
    "stream": true
  }'
```

#### 3. List Available Models

```bash
curl "http://localhost:8000/api/v1/models/list"
```

#### 4. Fine-tune a Model

```bash
curl -X POST "http://localhost:8000/api/v1/training/start" \
  -H "Content-Type: application/json" \
  -d '{
    "base_model": "gpt2",
    "provider": "huggingface",
    "dataset": {
      "texts": [
        "Training example 1",
        "Training example 2",
        "Training example 3"
      ]
    },
    "output_name": "my-custom-model",
    "epochs": 3,
    "batch_size": 8
  }'
```

#### 5. Check Training Status

```bash
curl "http://localhost:8000/api/v1/training/status/{job_id}"
```

## ⚙️ Configuration

All configuration is managed through environment variables or the `.env` file. See `.env.example` for all available options.

### Key Configuration Options

| Variable | Description | Default |
|----------|-------------|---------|
| `ENVIRONMENT` | Runtime environment | `development` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `ENABLED_PROVIDERS` | Comma-separated list of providers | `ollama,huggingface,openai` |
| `OLLAMA_BASE_URL` | Ollama service URL | `http://localhost:11434` |
| `HUGGINGFACE_CACHE_DIR` | HuggingFace models cache | `./models/huggingface` |
| `OPENAI_API_KEY` | OpenAI API key | Empty |
| `MAX_TOKENS` | Maximum generation tokens | `2048` |
| `DEFAULT_TEMPERATURE` | Default temperature | `0.7` |

## 🔧 Advanced Usage

### Adding a New Provider

1. Create a new provider class in `app/core/providers/`:

```python
from app.core.providers.base import BaseLLMProvider

class MyCustomProvider(BaseLLMProvider):
    async def initialize(self) -> None:
        # Initialize provider
        pass
    
    async def generate(self, request: GenerationRequest) -> GenerationResponse:
        # Implement generation
        pass
    
    # Implement other required methods...
```

2. Register the provider in `app/core/providers/__init__.py`:

```python
if "my_custom" in settings.enabled_providers:
    self.providers["my_custom"] = MyCustomProvider(config)
```

### Custom Training Pipeline

Extend the `TrainingManager` class to implement custom training logic:

```python
class CustomTrainingManager(TrainingManager):
    async def custom_train(self, job_id: str, custom_params):
        # Implement custom training logic
        pass
```

## 🧪 Testing

Run tests with pytest:

```bash
pytest tests/ -v
```

Run with coverage:

```bash
pytest tests/ --cov=app --cov-report=html
```

## 📊 Monitoring

The service provides several monitoring endpoints:

- `/health` - Basic health check
- `/api/v1/models/health` - Provider-specific health status
- Logs are output in JSON format for easy parsing

## 🚢 Production Deployment

### Best Practices

1. **Use environment-specific configurations**
   - Set `ENVIRONMENT=production`
   - Disable debug endpoints
   - Configure proper logging

2. **Enable GPU support**
   - Uncomment GPU configuration in `docker-compose.yml`
   - Ensure NVIDIA Docker runtime is installed

3. **Set up monitoring**
   - Integrate with Prometheus/Grafana
   - Configure log aggregation

4. **Secure the API**
   - Enable API key authentication
   - Use HTTPS with reverse proxy (nginx/traefik)
   - Implement rate limiting

5. **Scale horizontally**
   - Use container orchestration (Kubernetes)
   - Load balance multiple instances
   - Separate training and inference services

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- FastAPI framework
- HuggingFace Transformers
- Ollama project
- OpenAI API

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the documentation
- Review existing issues

---

**Built with ❤️ using FastAPI and modern Python**

## Authentication

If API key authentication is enabled, include the API key in the header:

```bash
curl -H "X-API-Key: your-api-key" http://localhost:8000/api/v1/models/list
```

## Text Generation Examples

### 1. Basic Generation (Ollama)

```python
import requests

url = "http://localhost:8000/api/v1/generation/generate"
payload = {
    "prompt": "Write a Python function to calculate fibonacci numbers",
    "provider": "ollama",
    "model": "codellama",
    "max_tokens": 500,
    "temperature": 0.7
}

response = requests.post(url, json=payload)
result = response.json()
print(result["generated_text"])
```

### 2. Streaming Generation

```python
import requests
import json

url = "http://localhost:8000/api/v1/generation/generate/stream"
payload = {
    "prompt": "Explain machine learning in detail",
    "provider": "ollama",
    "stream": True
}

with requests.post(url, json=payload, stream=True) as response:
    for line in response.iter_lines():
        if line:
            data = json.loads(line.decode('utf-8').replace('data: ', ''))
            if not data.get('done'):
                print(data['text'], end='', flush=True)
```

### 3. Chat with Context (OpenAI)

```python
import requests

url = "http://localhost:8000/api/v1/generation/generate"
payload = {
    "prompt": "What's the weather like?",
    "provider": "openai",
    "model": "gpt-3.5-turbo",
    "system_message": "You are a helpful weather assistant.",
    "context": [
        {"role": "user", "content": "I'm in New York"},
        {"role": "assistant", "content": "Got it! You're in New York."}
    ],
    "temperature": 0.7
}

response = requests.post(url, json=payload)
print(response.json()["generated_text"])
```

### 4. HuggingFace Local Model

```python
import requests

url = "http://localhost:8000/api/v1/generation/generate"
payload = {
    "prompt": "Once upon a time",
    "provider": "huggingface",
    "model": "gpt2",
    "max_tokens": 200,
    "temperature": 0.8,
    "top_p": 0.9
}

response = requests.post(url, json=payload)
print(response.json())
```

## Model Management Examples

### 1. List All Models

```python
import requests

response = requests.get("http://localhost:8000/api/v1/models/list")
models = response.json()

print(f"Total models: {models['total']}")
for model in models['models']:
    print(f"- {model['name']} ({model['provider']})")
```

### 2. List Provider-Specific Models

```python
import requests

providers = ["ollama", "huggingface", "openai"]

for provider in providers:
    response = requests.get(f"http://localhost:8000/api/v1/models/list/{provider}")
    if response.status_code == 200:
        models = response.json()
        print(f"\n{provider.upper()} Models:")
        for model in models:
            print(f"  - {model}")
```

### 3. Health Check

```python
import requests

response = requests.get("http://localhost:8000/api/v1/models/health")
health = response.json()

print(f"Overall Status: {health['status']}")
for provider, status in health['providers'].items():
    print(f"  {provider}: {'✓' if status else '✗'}")
```

## Model Training Examples

### 1. Fine-tune a Model

```python
import requests
import time

# Prepare training data
training_data = {
    "base_model": "gpt2",
    "provider": "huggingface",
    "dataset": {
        "texts": [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Python is a popular programming language.",
            # Add more training examples...
        ]
    },
    "output_name": "my-custom-gpt2",
    "epochs": 3,
    "batch_size": 4,
    "learning_rate": 2e-5
}

# Start training
response = requests.post(
    "http://localhost:8000/api/v1/training/start",
    json=training_data
)
job = response.json()
job_id = job['job_id']

print(f"Training started: {job_id}")

# Monitor progress
while True:
    response = requests.get(f"http://localhost:8000/api/v1/training/status/{job_id}")
    status = response.json()
    
    print(f"Status: {status['status']}, Progress: {status['progress']:.1f}%")
    
    if status['status'] in ['completed', 'failed']:
        break
    
    time.sleep(10)

if status['status'] == 'completed':
    print(f"Training completed! Model saved at: {status['model_path']}")
else:
    print(f"Training failed: {status['error']}")
```

### 2. Monitor Training Job

```python
import requests

job_id = "your-job-id-here"
response = requests.get(f"http://localhost:8000/api/v1/training/status/{job_id}")
status = response.json()

print(f"""
Training Job Status:
-------------------
Job ID: {status['job_id']}
Status: {status['status']}
Progress: {status['progress']:.1f}%
Current Epoch: {status.get('current_epoch', 'N/A')}
Total Epochs: {status.get('total_epochs', 'N/A')}
Loss: {status.get('loss', 'N/A')}
Started: {status.get('started_at', 'N/A')}
""")
```

### 3. List All Training Jobs

```python
import requests

response = requests.get("http://localhost:8000/api/v1/training/list")
data = response.json()

print(f"Total Training Jobs: {data['total']}\n")

for job in data['jobs']:
    print(f"Job ID: {job['job_id']}")
    print(f"Status: {job['status']}")
    print(f"Progress: {job['progress']:.1f}%")
    print("---")
```

### 4. Cancel Training Job

```python
import requests

job_id = "your-job-id-here"
response = requests.delete(f"http://localhost:8000/api/v1/training/cancel/{job_id}")

if response.status_code == 200:
    print("Training job cancelled successfully")
else:
    print(f"Failed to cancel job: {response.json()}")
```

## Advanced Usage

### Batch Processing

```python
import requests
from concurrent.futures import ThreadPoolExecutor

def generate_text(prompt):
    response = requests.post(
        "http://localhost:8000/api/v1/generation/generate",
        json={
            "prompt": prompt,
            "provider": "ollama",
            "max_tokens": 100
        }
    )
    return response.json()

prompts = [
    "Explain AI",
    "What is Python?",
    "Describe machine learning"
]

with ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(generate_text, prompts))

for prompt, result in zip(prompts, results):
    print(f"\nPrompt: {prompt}")
    print(f"Response: {result['generated_text'][:100]}...")
```

### Error Handling

```python
import requests
from requests.exceptions import RequestException

def safe_generate(prompt, max_retries=3):
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:8000/api/v1/generation/generate",
                json={"prompt": prompt, "provider": "ollama"},
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        except RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt == max_retries - 1:
                raise
    return None

result = safe_generate("Test prompt")
```