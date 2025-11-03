"""
OpenRouter API Client with Batched Generation and Semaphore Control
Simple wrapper for concurrent API calls with rate limiting
"""

import os
import asyncio
import aiohttp
from typing import List, Union, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ResponseResult:
    """Container for API response"""
    prompt: str
    response: str
    model: str
    error: Optional[str] = None
    
    @property
    def success(self) -> bool:
        return self.error is None


class Response:
    """OpenRouter API client with batched generation"""
    
    def __init__(
        self, 
        api_key: Optional[str] = None,
        max_concurrent: int = 5,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize OpenRouter client
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            max_concurrent: Maximum concurrent requests (semaphore limit)
            base_url: OpenRouter API base URL
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("API key required: pass api_key or set OPENROUTER_API_KEY env var")
        
        self.max_concurrent = max_concurrent
        self.base_url = base_url
        self.endpoint = f"{base_url}/chat/completions"
        
    def _format_messages(self, prompt: str) -> List[Dict[str, str]]:
        """Convert prompt to OpenAI-style messages format"""
        return [{"role": "user", "content": prompt}]
    
    async def _generate_single(
        self,
        session: aiohttp.ClientSession,
        semaphore: asyncio.Semaphore,
        prompt: str,
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> ResponseResult:
        """Generate response for a single prompt"""
        async with semaphore:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            
            payload = {
                "model": model,
                "messages": self._format_messages(prompt),
                "max_tokens": max_tokens,
                "temperature": temperature,
                **kwargs
            }
            
            try:
                async with session.post(
                    self.endpoint,
                    headers=headers,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=120)
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        return ResponseResult(
                            prompt=prompt,
                            response="",
                            model=model,
                            error=f"HTTP {resp.status}: {error_text}"
                        )
                    
                    data = await resp.json()
                    
                    # Check for API errors in response
                    if "error" in data:
                        return ResponseResult(
                            prompt=prompt,
                            response="",
                            model=model,
                            error=f"API Error: {data['error']}"
                        )
                    
                    # Extract response with validation
                    try:
                        response_text = data["choices"][0]["message"]["content"]
                        if response_text is None:
                            response_text = ""
                    except (KeyError, IndexError, TypeError) as e:
                        return ResponseResult(
                            prompt=prompt,
                            response="",
                            model=model,
                            error=f"Invalid response format: {e}. Response: {data}"
                        )
                    
                    return ResponseResult(
                        prompt=prompt,
                        response=response_text,
                        model=model
                    )
                    
            except Exception as e:
                return ResponseResult(
                    prompt=prompt,
                    response="",
                    model=model,
                    error=f"Exception: {str(e)}"
                )
    
    async def _generate_batch_async(
        self,
        prompts: List[str],
        model: str,
        max_tokens: int,
        temperature: float,
        **kwargs
    ) -> List[ResponseResult]:
        """Async batch generation with semaphore"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async with aiohttp.ClientSession() as session:
            tasks = [
                self._generate_single(
                    session, semaphore, prompt, model, max_tokens, temperature, **kwargs
                )
                for prompt in prompts
            ]
            results = await asyncio.gather(*tasks)
            
        return results
    
    def generate(
        self,
        prompt: str,
        model: str = "anthropic/claude-3.5-sonnet",
        max_tokens: int = 1000,
        temperature: float = 1.0,
        **kwargs
    ) -> str:
        """
        Generate response for a single prompt
        
        Args:
            prompt: Input prompt
            model: OpenRouter model name (e.g., "anthropic/claude-3.5-sonnet")
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional API parameters
            
        Returns:
            Generated response text
        """
        result = self.generate_batch(
            [prompt], model=model, max_tokens=max_tokens, temperature=temperature, **kwargs
        )[0]
        
        if not result.success:
            raise RuntimeError(f"Generation failed: {result.error}")
        
        return result.response
    
    def generate_batch(
        self,
        prompts: Union[str, List[str]],
        model: str = "anthropic/claude-3.5-sonnet",
        max_tokens: int = 1000,
        temperature: float = 1.0,
        **kwargs
    ) -> List[ResponseResult]:
        """
        Generate responses for multiple prompts with concurrency control
        
        Args:
            prompts: Single prompt or list of prompts
            model: OpenRouter model name
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional API parameters
            
        Returns:
            List of ResponseResult objects
        """
        # Normalize to list
        if isinstance(prompts, str):
            prompts = [prompts]
        
        # Run async batch generation
        results = asyncio.run(
            self._generate_batch_async(
                prompts, model, max_tokens, temperature, **kwargs
            )
        )
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize client
    client = Response(max_concurrent=5)
    
    # Single generation
    print("Single generation test:")
    response = client.generate(
        "What is 2+2?",
        model="anthropic/claude-3.5-sonnet",
        max_tokens=100
    )
    print(f"Response: {response}\n")
    
    # Batch generation
    print("Batch generation test:")
    prompts = [
        "What is the capital of France?",
        "What is the capital of Germany?",
        "What is the capital of Italy?",
        "What is the capital of Spain?",
        "What is the capital of Portugal?"
    ]
    
    results = client.generate_batch(
        prompts,
        model="openai/gpt-4o-mini",  # Valid model name
        max_tokens=50
    )
    
    for i, result in enumerate(results):
        if result.success:
            print(f"{i+1}. {result.prompt}")
            print(f"   → {result.response}\n")
        else:
            print(f"{i+1}. ERROR: {result.error}\n")

