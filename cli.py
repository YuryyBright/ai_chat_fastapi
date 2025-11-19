# cli.py
"""
Універсальний CLI для Local LLM Service (2025)
+ Повноцінний інтерактивний чат з збереженням контексту
+ Коректний стрімінг та вивід
"""

import click
import requests
import json
import time
from typing import Optional, Generator, List, Dict
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel
from rich.syntax import Syntax
from rich.markdown import Markdown
from rich.live import Live
from rich.text import Text

console = Console()


class LLMClient:
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})

    def _stream_sse(self, url: str, json_data: dict) -> Generator[dict, None, None]:
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
                        if not data or data == "[DONE]":
                            continue
                        try:
                            chunk = json.loads(data)
                            yield chunk
                        except json.JSONDecodeError:
                            # Іноді приходить не JSON (наприклад, OpenAI v1), ігноруємо
                            continue
        except requests.RequestException as e:
            console.print(f"[red]Помилка з'єднання:[/red] {e}")
            raise click.Abort()

    def generate_stream(self, prompt: str, model: Optional[str] = None, history: List[Dict] = None, **kwargs):
        payload = {
            "prompt": prompt,
            "model": model,
            "stream": True,
            "temperature": kwargs.get("temperature", 0.8),
            "max_tokens": kwargs.get("max_tokens", 2048),
            "stop_sequences": kwargs.get("stop_sequences"),
        }

        # Якщо є історія — додаємо як context
        if history:
            messages = []
            for msg in history:
                role = "assistant" if msg["role"] == "assistant" else "user"
                messages.append({"role": role, "content": msg["content"]})
            messages.append({"role": "user", "content": prompt})
            payload["context"] = messages
        else:
            payload["prompt"] = prompt

        return self._stream_sse(f"{self.base_url}/generate/stream", payload)

    def generate(self, prompt: str, model: Optional[str] = None, **kwargs):
        payload = {
            "prompt": prompt,
            "model": model,
            "temperature": kwargs.get("temperature", 0.8),
            "max_tokens": kwargs.get("max_tokens", 2048),
        }
        resp = self.session.post(f"{self.base_url}/generate", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()

    def list_models(self):
        resp = self.session.get(f"{self.base_url}/models/list")
        resp.raise_for_status()
        return resp.json()

    def health(self):
        try:
            r = self.session.get(f"{self.base_url}/health", timeout=5)
            r.raise_for_status()
            return r.json()
        except:
            return {"status": "down"}


@click.group()
@click.option('--url', '-u', default='http://localhost:8000', help='URL сервера')
@click.option('--model', '-m', default=None, help='Модель за замовчуванням')
@click.pass_context
def cli(ctx, url, model):
    ctx.obj = {"client": LLMClient(base_url=url), "default_model": model}


@cli.command()
@click.argument('prompt', nargs=-1, required=False)
@click.option('--model', '-m', help='Назва моделі')
@click.option('--temp', '-t', type=float, default=0.8)
@click.option('--max-tokens', type=int, default=2048)
@click.option('--no-stream', is_flag=True, help='Без стрімінгу')
@click.pass_obj
def ask(obj, prompt, model, temp, max_tokens, no_stream):
    """Одноразовий запит"""
    client = obj["client"]
    model = model or obj["default_model"]

    if not prompt:
        prompt = click.prompt("Промпт")
    else:
        prompt = " ".join(prompt)

    console.print(f"[bold cyan]Промпт:[/bold cyan] {prompt}\n")

    try:
        if no_stream:
            result = client.generate(prompt, model, temperature=temp, max_tokens=max_tokens)
            text = result.get("generated_text", "")
            console.print(f"[bold green]Відповідь:[/bold green]\n")
            console.print(Markdown(text))
        else:
            console.print("[bold green]Відповідь:[/bold green]")
            full = ""
            for chunk in client.generate_stream(prompt, model, temperature=temp, max_tokens=max_tokens):
                text = chunk.get("text", "")
                if text:
                    console.print(text, end="")
                    full += text
            console.print("\n")
    except Exception as e:
        console.print(f"[red]Помилка:[/red] {e}")
        raise click.Abort()


@cli.command()
@click.option('--model', '-m', help='Модель для чату')
@click.option('--temp', '-t', type=float, default=0.8)
@click.option('--system', '-s', default=None, help='Системне повідомлення')
@click.pass_obj
def chat(obj, model, temp, system):
    """Інтерактивний чат з пам’яттю"""
    client = obj["client"]
    model = model or obj["default_model"]

    if not model:
        # Автовизначення моделі
        try:
            models = client.list_models().get("models", [])
            if models:
                model = models[0]["name"]
                console.print(f"[dim]Використовується модель: {model}[/dim]")
            else:
                console.print("[red]Немає доступних моделей![/red]")
                return
        except:
            console.print("[red]Не вдалося отримати список моделей[/red]")
            return

    history = []
    if system:
        history.append({"role": "system", "content": system})

    console.print(Panel(
        f"[bold cyan]Чат з моделлю:[/bold cyan] {model}\n"
        f"[dim]Температура: {temp} | Введіть повідомлення (або /exit, /clear, /model)[/dim]",
        title="Local LLM Chat",
        border_style="bright_blue"
    ))

    while True:
        try:
            user_input = click.prompt("\n[bold yellow]Ти[/bold yellow]", default="", show_default=False)
            
            if user_input.strip() in {"", "/exit", "quit", "вихід"}:
                console.print("[dim]Бувай![/dim]")
                break
            if user_input.strip() == "/clear":
                history = [{"role": "system", "content": system or "Ти — корисний асистент."}]
                console.print("[dim]Історія очищена[/dim]")
                continue
            if user_input.strip() == "/model":
                console.print(f"[dim]Поточна модель: {model}[/dim]")
                continue

            history.append({"role": "user", "content": user_input})
            console.print(f"[bold green]Модель[/bold green]: ", end="")

            full_response = ""
            try:
                for chunk in client.generate_stream(
                    prompt=user_input,
                    model=model,
                    history=history[:-1],  # передаємо попередню історію
                    temperature=temp
                ):
                    text = chunk.get("text", "")
                    if text:
                        console.print(text, end="")
                        full_response += text
                console.print()  # новий рядок
            except KeyboardInterrupt:
                console.print("\n[dim]Перервано[/dim]")
                continue
            except Exception as e:
                console.print(f"\n[red]Помилка:[/red] {e}")
                continue

            if full_response.strip():
                history.append({"role": "assistant", "content": full_response})

        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]До побачення![/dim]")
            break


@cli.command()
@click.pass_obj
def models(obj):
    """Список моделей"""
    try:
        data = obj["client"].list_models()
        table = Table(title="Локальні моделі")
        table.add_column("Назва", style="cyan")
        table.add_column("Статус", style="green")
        table.add_column("Розмір", style="yellow")

        for m in data.get("models", []):
            meta = m.get("metadata", {})
            size = meta.get("size_mb", "?")
            if isinstance(size, (int, float)):
                size = f"{size:.1f} MB"
            table.add_row(m["name"], m["status"], str(size))

        console.print(table)
        console.print(f"[bold]Всього:[/bold] {len(data.get('models', []))}")
    except Exception as e:
        console.print(f"[red]Помилка:[/red] {e}")


@cli.command()
@click.pass_obj
def health(obj):
    data = obj["client"].health()
    status = data.get("status", "unknown")
    color = "green" if "healthy" in status.lower() else "red"
    console.print(f"Сервер: [{color}]{status.upper()}[/{color}]")


if __name__ == '__main__':
    cli()