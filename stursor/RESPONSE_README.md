# Response Class - OpenRouter API Client

Simple async OpenRouter API client with batched generation and semaphore control.

## Setup

1. Install dependencies:
```bash
pip install aiohttp
```

2. Set your API key:
```bash
export OPENROUTER_API_KEY='your-api-key-here'
```

Or in Python:
```python
import os
os.environ["OPENROUTER_API_KEY"] = "your-api-key-here"
```

## Quick Start

```python
from response import Response

# Initialize client
client = Response(max_concurrent=5)

# Single generation
response = client.generate(
    "What is AI?",
    model="anthropic/claude-3.5-sonnet",
    max_tokens=100
)
print(response)

# Batch generation
prompts = ["Question 1?", "Question 2?", "Question 3?"]
results = client.generate_batch(
    prompts,
    model="openai/gpt-4o-mini",
    max_tokens=50
)

for result in results:
    if result.success:
        print(f"Q: {result.prompt}")
        print(f"A: {result.response}\n")
    else:
        print(f"Error: {result.error}")
```

## API Reference

### `Response(api_key=None, max_concurrent=5, base_url="https://openrouter.ai/api/v1")`

Initialize the OpenRouter client.

**Parameters:**
- `api_key` (str, optional): OpenRouter API key. Defaults to `OPENROUTER_API_KEY` env var.
- `max_concurrent` (int): Maximum concurrent API requests (semaphore limit). Default: 5.
- `base_url` (str): OpenRouter API base URL.

---

### `generate(prompt, model, max_tokens=1000, temperature=1.0, **kwargs)`

Generate response for a single prompt.

**Parameters:**
- `prompt` (str): Input prompt
- `model` (str): OpenRouter model name (e.g., `"anthropic/claude-3.5-sonnet"`)
- `max_tokens` (int): Maximum tokens to generate. Default: 1000.
- `temperature` (float): Sampling temperature (0.0 to 2.0). Default: 1.0.
- `**kwargs`: Additional API parameters

**Returns:** `str` - Generated response text

**Raises:** `RuntimeError` if generation fails

---

### `generate_batch(prompts, model, max_tokens=1000, temperature=1.0, **kwargs)`

Generate responses for multiple prompts with concurrency control.

**Parameters:**
- `prompts` (str or List[str]): Single prompt or list of prompts
- `model` (str): OpenRouter model name
- `max_tokens` (int): Maximum tokens to generate. Default: 1000.
- `temperature` (float): Sampling temperature. Default: 1.0.
- `**kwargs`: Additional API parameters

**Returns:** `List[ResponseResult]` - List of result objects

---

### `ResponseResult`

Result object containing:
- `prompt` (str): Original prompt
- `response` (str): Generated response
- `model` (str): Model used
- `error` (str or None): Error message if failed
- `success` (bool): Property indicating if generation succeeded

## Supported Models

Popular models on OpenRouter:

**Anthropic:**
- `anthropic/claude-3.5-sonnet`
- `anthropic/claude-3-opus`
- `anthropic/claude-3-haiku`

**OpenAI:**
- `openai/gpt-4o`
- `openai/gpt-4o-mini`
- `openai/gpt-4-turbo`

**Google:**
- `google/gemini-2.0-flash-exp:free`
- `google/gemini-pro`

**Meta:**
- `meta-llama/llama-3.3-70b-instruct`

See full list at: https://openrouter.ai/models

## Examples

### Different temperatures
```python
# Creative (high temperature)
response = client.generate(
    "Write a story",
    model="anthropic/claude-3.5-sonnet",
    temperature=1.5,
    max_tokens=500
)

# Factual (low temperature)
response = client.generate(
    "What is 2+2?",
    model="openai/gpt-4o-mini",
    temperature=0.0,
    max_tokens=10
)
```

### Large batch processing
```python
# Process 100 prompts with semaphore control
prompts = [f"Question {i}" for i in range(100)]
results = client.generate_batch(
    prompts,
    model="openai/gpt-4o-mini",
    max_tokens=50
)

# Filter successful results
successful = [r for r in results if r.success]
print(f"Completed {len(successful)}/{len(prompts)}")
```

### Error handling
```python
results = client.generate_batch(prompts, model="some-model")

for result in results:
    if result.success:
        print(result.response)
    else:
        print(f"Failed: {result.error}")
```

## Demo Script

Run the demo:
```bash
python response_demo.py
```

This will show:
1. Single generation
2. Batch generation
3. Different models
4. Large batch with semaphore control

## Notes

- The semaphore controls concurrent API requests to avoid rate limits
- All requests are async for maximum throughput
- Failed requests return `ResponseResult` with `error` set (no retries)
- The client handles message formatting automatically

