# cli.py
"""
Оновлений CLI для Local LLM Service (2025)
Повністю сумісний з новим API v2
"""

import click
import requests
import json
import time
from typing import Optional, Generator
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax

console = Console()


class LLMClient:
    """Клієнт для взаємодії з Local LLM Service API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _stream_sse(self, url: str, json_data: dict) -> Generator[dict, None, None]:
        """Універсальний генератор для Server-Sent Events (SSE)"""
        with self.session.post(url, json=json_data, stream=True, timeout=3600) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    decoded = line.decode("utf-8").strip()
                    if decoded.startswith("data: "):
                        data_str = decoded[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            yield json.loads(data_str)
                        except json.JSONDecodeError:
                            console.print(f"[dim]Не вдалося розпарсити: {data_str}[/dim]")

    def generate(self, prompt: str, model: Optional[str] = None, stream: bool = True, **kwargs):
        url = f"{self.base_url}/generate/stream" if stream else f"{self.base_url}/generate"
        payload = {
            "prompt": prompt,
            "model": model,
            **kwargs
        }
        if stream:
            return self._stream_sse(url, payload)
        else:
            resp = self.session.post(url, json=payload)
            resp.raise_for_status()
            return resp.json()

    def list_models(self):
        resp = self.session.get(f"{self.base_url}/models/list")
        resp.raise_for_status()
        return resp.json()

    def download_model(self, repo_id: str, filename: Optional[str] = None, model_type: str = "gguf"):
        url = f"{self.base_url}/models/download"
        payload = {
            "repo_id": repo_id,
            "model_type": model_type,
            "filename": filename
        }
        return self._stream_sse(url, payload)

    def search_models(self, query: str = "", limit: int = 20):
        resp = self.session.post(
            f"{self.base_url}/models/search",
            json={"query": query, "limit": limit}
        )
        resp.raise_for_status()
        return resp.json()

    def list_repo_files(self, repo_id: str):
        resp = self.session.get(f"{self.base_url}/models/files/{repo_id}")
        resp.raise_for_status()
        return resp.json()

    def delete_model(self, model_name: str):
        resp = self.session.delete(f"{self.base_url}/models/delete", json={"model_name": model_name})
        resp.raise_for_status()
        return resp.json()

    def model_info(self, model_name: str):
        resp = self.session.get(f"{self.base_url}/models/info/{model_name}")
        resp.raise_for_status()
        return resp.json()

    def health(self):
        resp = self.session.get(f"{self.base_url}/health")
        resp.raise_for_status()
        return resp.json()


@click.group()
@click.option('--url', '-u', default='http://localhost:8000', help='Базова URL API (за замовчуванням: http://localhost:8000)')
@click.pass_context
def cli(ctx, url):
    """Local LLM Service CLI — керування локальними моделями та генерація тексту"""
    ctx.obj = LLMClient(base_url=url)


@cli.command()
@click.argument('prompt', nargs=-1)
@click.option('--model', '-m', help='Назва моделі (якщо не вказано — перша доступна)')
@click.option('--temp', '-t', 'temperature', type=float, default=0.7, help='Температура (0.0–2.0)')
@click.option('--max-tokens', type=int, default=1024, help='Максимум токенів')
@click.option('--no-stream', is_flag=True, help='Вимкнути потокову відповідь')
@click.pass_obj
def ask(client: LLMClient, prompt, model, temperature, max_tokens, no_stream):
    """Запитати модель (інтерактивно або одноразово)"""
    if not prompt:
        click.echo("Введіть промпт:")
        prompt = click.get_text_stream('stdin').readline().rstrip()
    else:
        prompt = " ".join(prompt)

    console.print(f"[bold cyan]Промпт:[/bold cyan] {prompt}\n")

    if no_stream:
        with Progress(SpinnerColumn(), TextColumn("Генерація..."), console=console) as progress:
            task = progress.add_task("generate")
            result = client.generate(prompt, model, stream=False,
                                    temperature=temperature, max_tokens=max_tokens)
            progress.remove_task(task)
        console.print(f"[bold green]Відповідь:[/bold green]\n{result['generated_text']}")
    else:
        console.print("[bold green]Відповідь:[/bold green] ", end="")
        for chunk in client.generate(prompt, model, stream=True,
                                   temperature=temperature, max_tokens=max_tokens):
            if "text" in chunk:
                console.print(chunk["text"], end="")
        console.print("\n")


@cli.command()
@click.pass_obj
def models(client: LLMClient):
    """Показати список локальних моделей"""
    data = client.list_models()
    table = Table(title="Локальні моделі")
    table.add_column("Назва", style="cyan")
    table.add_column("Статус", style="green")
    table.add_column("Розмір", style="yellow")

    for model in data["models"]:
        size = model["metadata"].get("size", "невідомо") if model["metadata"] else "невідомо"
        table.add_row(model["name"], model["status"], size)

    console.print(table)
    console.print(f"[bold]Всього:[/bold] {data['total']} модел{'ь' if data['total'] != 1 else 'і'}")


@cli.command()
@click.argument('repo_id')
@click.option('--filename', '-f', help='Конкретний GGUF файл (наприклад: model-q5_k.gguf)')
@click.option('--type', 'model_type', type=click.Choice(['gguf', 'huggingface', 'gptq', 'awq', 'exl2']), default='gguf')
@click.pass_obj
def download(client: LLMClient, repo_id, filename, model_type):
    """Завантажити модель з HuggingFace"""
    if model_type == "gguf" and not filename:
        # Спробуємо автоматично отримати список файлів
        console.print("[yellow]Файл не вказано — отримуємо список GGUF...[/yellow]")
        files = client.list_repo_files(repo_id).get("files", [])
        gguf_files = [f for f in files if f.lower().endswith(".gguf")]
        if not gguf_files:
            console.print("[red]Не знайдено GGUF файлів у репозиторії[/red]")
            raise click.Abort()
        # Пропонуємо вибрати
        console.print("Знайдено файли:")
        for i, f in enumerate(gguf_files):
            console.print(f"  {i+1}. {f}")
        choice = click.prompt("Виберіть номер файлу", type=int)
        filename = gguf_files[choice - 1]

    console.print(f"[bold yellow]Завантаження {repo_id} → {filename or model_type}[/bold yellow]")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Завантаження...", total=100)

        for event in client.download_model(repo_id, filename, model_type):
            if event["status"] == "downloading":
                progress.update(task, completed=event["progress"])
            elif event["status"] == "complete":
                progress.update(task, completed=100)
                console.print(f"[bold green]Готово![/bold green] Модель завантажена: {event.get('model_path', '')}")
            elif event["status"] == "error":
                console.print(f"[bold red]Помилка:[/bold red] {event['message']}")
                raise click.Abort()


@cli.command()
@click.argument('query', required=False)
@click.option('--limit', '-l', default=15, help='Кількість результатів')
@click.pass_obj
def search(client: LLMClient, query, limit):
    """Пошук моделей на HuggingFace (GGUF)"""
    query = query or ""
    results = client.search_models(query, limit)
    table = Table(title=f"Результати пошуку: '{query}'")
    table.add_column("Автор", style="cyan")
    table.add_column("ID", style="green")
    table.add_column("Завантаження", justify="right")
    table.add_column("Лайки", justify="right")

    for model in results.get("models", []):
        author = model["id"].split("/")[0]
        name = model["id"].split("/")[-1]
        table.add_row(author, name, str(model.get("downloads", 0)), str(model.get("likes", 0)))

    console.print(table)


@cli.command()
@click.argument('model_name')
@click.pass_obj
def info(client: LLMClient, model_name):
    """Інформація про локальну модель"""
    data = client.model_info(model_name)
    info = data["model"]
    panel = Panel(
        Syntax(json.dumps(info, indent=2, ensure_ascii=False), "json", theme="monokai", line_numbers=True),
        title=f"Модель: [bold cyan]{model_name}[/bold cyan]",
        border_style="bright_blue"
    )
    console.print(panel)


@cli.command()
@click.argument('model_name')
@click.pass_obj
def delete(client: LLMClient, model_name):
    """Видалити локальну модель"""
    if not click.confirm(f"Видалити модель '{model_name}'?"):
        console.print("Скасовано.")
        return
    result = client.delete_model(model_name)
    console.print(f"[bold green]Успішно видалено:[/bold green] {model_name}")


@cli.command()
@click.pass_obj
def health(client: LLMClient):
    """Перевірка стану сервісу"""
    data = client.health()
    status = "Здоровий" if data["status"] == "healthy" else "Проблеми"
    color = "green" if data["status"] == "healthy" else "red"
    console.print(f"Статус: [{color}]{status}[/{color}]")
    console.print(f"Версія: {data['version']}")


if __name__ == '__main__':
    cli()