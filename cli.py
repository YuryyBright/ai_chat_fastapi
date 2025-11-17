# cli.py
"""
Command-line interface for LLM Service
Provides convenient CLI commands for interacting with the service
"""

import click
import requests
import json
from typing import Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, DownloadColumn, TransferSpeedColumn, TimeRemainingColumn

console = Console()


class LLMClient:
    """Client for interacting with LLM Service API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
    
    def generate(
        self,
        prompt: str,
        provider: str = "ollama",
        model: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ):
        """Generate text"""
        url = f"{self.base_url}/api/v1/generation/generate"
        if stream:
            url = f"{self.base_url}/api/v1/generation/generate/stream"
        
        payload = {
            "prompt": prompt,
            "provider": provider,
            "model": model,
            "stream": stream,
            **kwargs
        }
        
        if stream:
            with requests.post(url, json=payload, stream=True) as response:
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8').replace('data: ', ''))
                        if not data.get('done'):
                            yield data['text']
        else:
            response = requests.post(url, json=payload)
            response.raise_for_status()
            return response.json()
    
    def list_models(self, provider: Optional[str] = None):
        """List available models"""
        if provider:
            url = f"{self.base_url}/api/v1/models/list/{provider}"
        else:
            url = f"{self.base_url}/api/v1/models/list"
        
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    
    def health_check(self):
        """Check service health"""
        url = f"{self.base_url}/api/v1/models/health"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def pull_model(self, model_name: str, provider: str = "huggingface"):
        """Download model from Hugging Face"""
        url = f"{self.base_url}/api/v1/models/pull"
        payload = {
            "model_name": model_name,
            "provider": provider
        }
        
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            total = None
            downloaded = 0
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                DownloadColumn(),
                TransferSpeedColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task(f"[cyan]Downloading {model_name}...", total=100)
                
                for line in response.iter_lines():
                    if line:
                        data = json.loads(line.decode('utf-8').lstrip('data: ').strip())
                        
                        if data.get("status") == "progress":
                            progress.update(task, completed=data["progress"])
                        elif data.get("status") == "complete":
                            progress.update(task, completed=100)
                            console.print(f"\n[bold green]Model '{model_name}' successfully downloaded and cached![/bold green]")
                        elif data.get("status") == "error":
                            console.print(f"\n[bold red]Error: {data.get('error')}[/bold red]")
                            return


@click.group()
@click.option('--url', default='http://localhost:8000', help='API base URL')
@click.pass_context
def cli(ctx, url):
    """LLM Service CLI - Interact with the LLM service from command line"""
    ctx.obj = LLMClient(base_url=url)


# === Існуючі команди (generate, models, health, batch) без змін ===
@cli.command()
@click.argument('prompt')
@click.option('--provider', '-p', default='ollama', help='LLM provider')
@click.option('--model', '-m', default=None, help='Model name')
@click.option('--stream/--no-stream', default=False, help='Enable streaming')
@click.option('--temperature', '-t', default=0.7, type=float, help='Temperature')
@click.option('--max-tokens', default=500, type=int, help='Max tokens')
@click.pass_obj
def generate(client, prompt, provider, model, stream, temperature, max_tokens):
    """Generate text from a prompt"""
    console.print(f"\n[bold cyan]Prompt:[/bold cyan] {prompt}\n")
    
    if stream:
        console.print("[bold green]Response:[/bold green] ", end="")
        for chunk in client.generate(
            prompt=prompt,
            provider=provider,
            model=model,
            stream=True,
            temperature=temperature,
            max_tokens=max_tokens
        ):
            console.print(chunk, end="")
        console.print("\n")
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            task = progress.add_task("Generating...", total=None)
            result = client.generate(
                prompt=prompt,
                provider=provider,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens
            )
            progress.remove_task(task)
        
        console.print(f"\n[bold green]Response:[/bold green]\n{result['generated_text']}\n")
        console.print(f"[dim]Model: {result['model']} | "
                     f"Time: {result['generation_time']:.2f}s | "
                     f"Tokens: {result.get('tokens_used', 'N/A')}[/dim]")


@cli.command()
@click.option('--provider', '-p', default=None, help='Filter by provider')
@click.pass_obj
def models(client, provider):
    """List available models"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Fetching models...", total=None)
        models_data = client.list_models(provider)
        progress.remove_task(task)
    
    if provider:
        table = Table(title=f"{provider.upper()} Models")
        table.add_column("Model Name", style="cyan")
        for model in models_data:
            table.add_row(model)
    else:
        table = Table(title="Available Models")
        table.add_column("Model Name", style="cyan")
        table.add_column("Provider", style="magenta")
        table.add_column("Type", style="green")
        table.add_column("Status", style="yellow")
        
        for model in models_data['models']:
            table.add_row(
                model['name'],
                model['provider'],
                model['type'],
                model['status']
            )
    
    console.print(table)


@cli.command()
@click.pass_obj
def health(client):
    """Check service health"""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        task = progress.add_task("Checking health...", total=None)
        health_data = client.health_check()
        progress.remove_task(task)
    
    table = Table(title="Service Health")
    table.add_column("Provider", style="cyan")
    table.add_column("Status", style="bold")
    
    for provider, status in health_data['providers'].items():
        status_text = "Available" if status else "Unavailable"
        status_style = "green" if status else "red"
        table.add_row(provider, f"[{status_style}]{status_text}[/{status_style}]")
    
    console.print(table)
    overall = "[green]Healthy[/green]" if health_data['status'] == "healthy" else "[yellow]Degraded[/yellow]"
    console.print(f"\nOverall Status: {overall}")


@cli.command()
@click.argument('input_file', type=click.File('r'))
@click.option('--output', '-o', type=click.File('w'), help='Output file')
@click.option('--provider', '-p', default='ollama', help='LLM provider')
@click.option('--model', '-m', default=None, help='Model name')
@click.pass_obj
def batch(client, input_file, output, provider, model):
    """Process prompts from a file"""
    prompts = [line.strip() for line in input_file if line.strip()]
    results = []
    
    with Progress(console=console) as progress:
        task = progress.add_task("[cyan]Processing prompts...", total=len(prompts))
        
        for prompt in prompts:
            result = client.generate(prompt=prompt, provider=provider, model=model)
            results.append({"prompt": prompt, "response": result['generated_text']})
            progress.update(task, advance=1)
    
    if output:
        json.dump(results, output, indent=2, ensure_ascii=False)
        console.print(f"\n[green]Results saved to {output.name}[/green]")
    else:
        for i, result in enumerate(results, 1):
            console.print(f"\n[bold cyan]Prompt {i}:[/bold cyan] {result['prompt']}")
            console.print(f"[bold green]Response:[/bold green] {result['response']}\n")


# === НОВА КОМАНДА: Завантаження моделі з Hugging Face ===
@cli.command()
@click.argument('model_name')
@click.option('--provider', '-p', default='huggingface', help='Provider (huggingface only for now)')
@click.pass_obj
def pull(client, model_name, provider):
    """Download a model from Hugging Face Hub
    
    Example:
        llm pull microsoft/DialoGPT-medium
        llm pull gpt2
        llm pull google/flan-t5-base --provider huggingface
    """
    if provider != "huggingface":
        console.print(f"[red]Pull command currently supports only 'huggingface' provider[/red]")
        raise click.Abort()
    
    console.print(f"[bold yellow]Starting download of {model_name} from Hugging Face...[/bold yellow]")
    try:
        client.pull_model(model_name, provider)
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            console.print(f"[bold red]Model '{model_name}' not found or endpoint not available[/bold red]")
        else:
            console.print(f"[bold red]HTTP Error: {e.response.status_code} - {e.response.text}[/bold red]")
    except Exception as e:
        console.print(f"[bold red]Failed to download model: {e}[/bold red]")


if __name__ == '__main__':
    cli()