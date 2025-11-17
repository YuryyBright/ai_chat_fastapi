# scripts/benchmark.py
"""
Benchmark script for comparing different models and providers
"""

import time
import requests
import statistics
from typing import List, Dict
from tabulate import tabulate


def benchmark_generation(
    base_url: str,
    prompt: str,
    configs: List[Dict],
    iterations: int = 5
) -> List[Dict]:
    """
    Benchmark text generation across different configurations
    
    Args:
        base_url: API base URL
        prompt: Test prompt
        configs: List of configuration dicts with provider and model
        iterations: Number of iterations per config
    
    Returns:
        List of benchmark results
    """
    results = []
    
    for config in configs:
        times = []
        tokens = []
        
        print(f"\nTesting {config['provider']} - {config.get('model', 'default')}...")
        
        for i in range(iterations):
            try:
                response = requests.post(
                    f"{base_url}/api/v1/generation/generate",
                    json={
                        "prompt": prompt,
                        "provider": config['provider'],
                        "model": config.get('model'),
                        "max_tokens": 100
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    data = response.json()
                    times.append(data['generation_time'])
                    if data.get('tokens_used'):
                        tokens.append(data['tokens_used'])
                    print(f"  Iteration {i+1}: {data['generation_time']:.2f}s")
                else:
                    print(f"  Iteration {i+1}: Failed ({response.status_code})")
            
            except Exception as e:
                print(f"  Iteration {i+1}: Error - {e}")
        
        if times:
            result = {
                'provider': config['provider'],
                'model': config.get('model', 'default'),
                'avg_time': statistics.mean(times),
                'min_time': min(times),
                'max_time': max(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0,
                'avg_tokens': statistics.mean(tokens) if tokens else None,
                'success_rate': (len(times) / iterations) * 100
            }
            results.append(result)
    
    return results


def print_benchmark_results(results: List[Dict]):
    """Print benchmark results in a formatted table"""
    
    table_data = []
    for result in results:
        table_data.append([
            f"{result['provider']}/{result['model']}",
            f"{result['avg_time']:.2f}s",
            f"{result['min_time']:.2f}s",
            f"{result['max_time']:.2f}s",
            f"{result['std_dev']:.2f}s",
            f"{result['avg_tokens']:.0f}" if result['avg_tokens'] else "N/A",
            f"{result['success_rate']:.0f}%"
        ])
    
    headers = [
        "Provider/Model",
        "Avg Time",
        "Min Time",
        "Max Time",
        "Std Dev",
        "Avg Tokens",
        "Success Rate"
    ]
    
    print("\n" + "="*80)
    print("BENCHMARK RESULTS")
    print("="*80)
    print(tabulate(table_data, headers=headers, tablefmt="grid"))


if __name__ == "__main__":
    base_url = "http://localhost:8000"
    test_prompt = "Explain what artificial intelligence is in 2-3 sentences."
    
    # Configure providers to test
    configs = [
        {"provider": "ollama", "model": "llama2"},
        {"provider": "huggingface", "model": "gpt2"},
        # Add more configurations as needed
    ]
    
    print("Starting benchmark...")
    print(f"Prompt: {test_prompt}")
    print(f"Iterations per config: 5")
    
    results = benchmark_generation(base_url, test_prompt, configs)
    print_benchmark_results(results)

