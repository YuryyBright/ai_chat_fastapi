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

```
llm-service/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application entry point
│   ├── config.py               # Pydantic configuration settings
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   └── routes/
│   │       ├── __init__.py
│   │       ├── generation.py   # Text generation endpoints
│   │       ├── models.py       # Model management endpoints
│   │       └── training.py     # Training endpoints
│   │
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exceptions.py       # Custom exceptions
│   │   ├── training.py         # Training manager
│   │   └── providers/
│   │       ├── __init__.py     # Provider manager
│   │       ├── base.py         # Base provider interface
│   │       ├── ollama_provider.py
│   │       ├── huggingface_provider.py
│   │       └── openai_provider.py
│   │
│   └── schemas/
│       ├── __init__.py
│       ├── generation.py       # Generation schemas
│       ├── models.py           # Model schemas
│       └── training.py         # Training schemas
│
├── data/                       # Data storage
├── models/                     # Model storage
│   ├── huggingface/           # HuggingFace cache
│   └── fine-tuned/            # Fine-tuned models
│
├── tests/                      # Test files
│   ├── __init__.py
│   ├── test_generation.py
│   ├── test_models.py
│   └── test_training.py
│
├── Dockerfile                  # Docker configuration
├── docker-compose.yml         # Docker Compose configuration
├── requirements.txt           # Python dependencies
├── .env.example               # Environment variables example
├── .dockerignore
├── .gitignore
└── README.md
```

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