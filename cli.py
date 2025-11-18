# cli.py
"""
Оновлений CLI для Local LLM Service (2025)
Виправлено парсинг відповідей та стрімінг
"""

import click
import requests
import json
import time
from typing import Optional, Generator, Any
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown

console = Console()


class LLMClient:
    """Універсальний клієнт для Local LLM Service (підтримує v1 та v2 API)"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.version = self._detect_version()

    def _detect_version(self) -> str:
        """Автовизначення версії API"""
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=3)
            if r.status_code == 200:
                data = r.json()
                if "version" in data:
                    return data.get("version", "v2")
        except:
            pass
        return "v2"  # fallback

    def _stream_sse(self, url: str, json_data: dict) -> Generator[dict, None, None]:
        """Правильний парсер Server-Sent Events"""
        try:
            with self.session.post(url, json=json_data, stream=True, timeout=3600) as r:
                r.raise_for_status()
                buffer = ""
                
                for line in r.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    
                    line = line.strip()
                    
                    if line.startswith("data: "):
                        data = line[6:].strip()
                        
                        if data in ("[DONE]", "[DONE]\n"):
                            break
                        
                        if data == "":
                            continue
                        
                        try:
                            yield json.loads(data)
                        except json.JSONDecodeError as e:
                            console.print(f"[dim red]JSON помилка: {data[:100]}[/dim red]")
                            
        except requests.exceptions.RequestException as e:
            console.print(f"[red]Помилка з'єднання:[/red] {e}")
            raise click.Abort()

    def generate(self, prompt: str, model: Optional[str] = None, stream: bool = True, **kwargs):
        """Генерація тексту — автоматично вибирає правильний ендпоінт"""
        if self.version.startswith("v1"):
            url = f"{self.base_url}/v1/chat/completions"
            payload = {
                "model": model or "local",
                "messages": [{"role": "user", "content": prompt}],
                "stream": stream,
                **{k: v for k, v in kwargs.items() if v is not None}
            }
        else:
            url = f"{self.base_url}/generate/stream" if stream else f"{self.base_url}/generate"
            payload = {
                "prompt": prompt,
                "model": model,
                **kwargs
            }

        if not stream:
            resp = self.session.post(url, json=payload, timeout=300)
            resp.raise_for_status()
            return resp.json()

        return self._stream_sse(url, payload)

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
            json={"query": query, "limit": limit},
            timeout=60
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
        try:
            resp = self.session.get(f"{self.base_url}/health", timeout=5)
            resp.raise_for_status()
            return resp.json()
        except:
            return {"status": "unavailable", "provider": "unknown"}


@click.group()
@click.option('--url', '-u', default='http://localhost:8000', help='URL сервера')
@click.pass_context
def cli(ctx, url):
    """Local LLM Service CLI — повне керування локальними моделями"""
    ctx.obj = LLMClient(base_url=url)


@cli.command()
@click.argument('prompt', nargs=-1, required=False)
@click.option('--model', '-m', help='Назва моделі')
@click.option('--temp', '-t', type=float, default=0.8)
@click.option('--max-tokens', type=int, default=2048)
@click.option('--no-stream', is_flag=True, help='Без потокової передачі')
@click.pass_obj
def ask(client: LLMClient, prompt, model, temp, max_tokens, no_stream):
    """Запит до моделі"""
    if not prompt:
        prompt = click.prompt("Промпт", type=str)
    else:
        prompt = " ".join(prompt)

    console.print(f"[bold cyan]Промпт:[/bold cyan] {prompt}\n")

    try:
        if no_stream:
            with Progress(SpinnerColumn(), TextColumn("Генерація..."), console=console) as p:
                p.add_task("gen")
                result = client.generate(prompt, model, stream=False,
                                        temperature=temp, max_tokens=max_tokens)
            
            # Парсинг відповіді v2 API
            text = result.get("generated_text", "")
            
            console.print(f"\n[bold green]Відповідь:[/bold green]\n")
            console.print(Markdown(text))
        else:
            console.print("[bold green]Відповідь:[/bold green]\n")
            full = ""
            
            for chunk in client.generate(prompt, model, stream=True,
                                       temperature=temp, max_tokens=max_tokens):
                # Парсинг v2 API стрімінгу
                text = chunk.get("text", "")
                done = chunk.get("done", False)
                error = chunk.get("error")
                
                if error:
                    console.print(f"\n[red]Помилка:[/red] {error}")
                    break
                
                if text:
                    console.print(text, end="")
                    full += text
                
                if done:
                    break
            
            console.print("\n")
            
    except Exception as e:
        console.print(f"[red]Помилка генерації:[/red] {e}")
        raise click.Abort()


@cli.command()
@click.pass_obj
def models(client: LLMClient):
    """Список локальних моделей"""
    try:
        data = client.list_models()
        table = Table(title="Локальні моделі")
        table.add_column("Назва", style="cyan")
        table.add_column("Статус", style="green")
        table.add_column("Розмір", style="yellow")
        table.add_column("Тип", style="magenta")

        for m in data.get("models", []):
            meta = m.get("metadata", {})
            size = meta.get("size_mb", meta.get("size", "невідомо"))
            if isinstance(size, (int, float)):
                size = f"{size} MB"
            typ = meta.get("type", "gguf")
            table.add_row(m["name"], m["status"], str(size), typ)

        console.print(table)
        console.print(f"[bold]Всього:[/bold] {data.get('total', 0)}")
    except Exception as e:
        console.print(f"[red]Не вдалося отримати список моделей:[/red] {e}")


@cli.command()
@click.argument('repo_id')
@click.option('--filename', '-f', help='GGUF файл (наприклад: model-q5_k_m.gguf)')
@click.option('--type', 'model_type', type=click.Choice(['gguf', 'huggingface']), default='gguf')
@click.pass_obj
def download(client: LLMClient, repo_id, filename, model_type):
    """Завантажити модель з HuggingFace"""
    if model_type == "gguf" and not filename:
        console.print("[yellow]Отримуємо список GGUF файлів...[/yellow]")
        try:
            files = client.list_repo_files(repo_id).get("files", [])
            ggufs = [f for f in files if f.lower().endswith(".gguf")]
            if not ggufs:
                console.print("[red]GGUF файли не знайдено[/red]")
                raise click.Abort()
            for i, f in enumerate(ggufs[:10], 1):
                console.print(f"  {i}. {f}")
            choice = click.prompt("Виберіть файл (номер)", type=int, default=1)
            filename = ggufs[choice - 1]
        except Exception as e:
            console.print(f"[red]Не вдалося отримати файли:[/red] {e}")
            raise click.Abort()

    console.print(f"[bold yellow]Завантажується:[/bold yellow] {repo_id} → {filename or 'весь репозиторій'}")

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TimeRemainingColumn(),
        console=console
    ) as progress:
        task = progress.add_task("Завантаження", total=100)
        try:
            for event in client.download_model(repo_id, filename, model_type):
                if event.get("progress") is not None:
                    progress.update(task, completed=event["progress"])
                if event["status"] == "complete":
                    progress.update(task, completed=100)
                    console.print(f"[bold green]Готово![/bold green] {event.get('model_path', '')}")
                elif event["status"] == "error":
                    console.print(f"[bold red]Помилка:[/bold red] {event.get('message')}")
        except Exception as e:
            console.print(f"[red]Перервано:[/red] {e}")


@cli.command()
@click.argument('query', required=False)
@click.option('--limit', '-l', default=15)
@click.pass_obj
def search(client: LLMClient, query, limit):
    """Пошук GGUF моделей на HF"""
    query = query or ""
    with Progress(SpinnerColumn(), TextColumn("Пошук..."), console=console) as p:
        p.add_task("search")
        results = client.search_models(query, limit)
    table = Table(title=f"Знайдено за запитом: [bold]'{query}'[/bold]")
    table.add_column("ID", style="green")
    table.add_column("Завантаження", justify="right")
    table.add_column("Лайки", justify="right")
    for m in results.get("models", []):
        table.add_row(m["id"], str(m.get("downloads", 0)), str(m.get("likes", 0)))
    console.print(table)


@cli.command()
@click.argument('model_name')
@click.pass_obj
def info(client: LLMClient, model_name):
    """Детальна інформація про модель"""
    try:
        data = client.model_info(model_name)
        info = data["model"]
        console.print(Panel(
            Syntax(json.dumps(info, indent=2, ensure_ascii=False), "json", theme="monokai"),
            title=f"[bold cyan]{model_name}[/bold cyan]",
            border_style="bright_blue"
        ))
    except Exception as e:
        console.print(f"[red]Модель не знайдена:[/red] {e}")


@cli.command()
@click.argument('model_name')
@click.confirmation_option(prompt='Видалити модель назавжди?')
@click.pass_obj
def delete(client: LLMClient, model_name):
    """Видалити модель"""
    try:
        client.delete_model(model_name)
        console.print(f"[bold green]Видалено:[/bold green] {model_name}")
    except Exception as e:
        console.print(f"[red]Помилка видалення:[/red] {e}")


@cli.command()
@click.pass_obj
def health(client: LLMClient):
    """Стан сервера"""
    data = client.health()
    status = data.get("status", "unknown")
    color = "green" if status == "healthy" else "red"
    console.print(f"Статус: [{color}]{status.upper()}[/{color}]")
    if "provider" in data:
        console.print(f"Провайдер: {data['provider']}")


if __name__ == '__main__':
    cli()