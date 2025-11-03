#!/usr/bin/env python3
"""
Demo script for Response class - OpenRouter API client
"""

import os
from response import Response


def main():
    print("="*60)
    print("OPENROUTER API CLIENT DEMO")
    print("="*60)
    print()
    
    # Check for API key
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: OPENROUTER_API_KEY not set!")
        print("   Set it with: export OPENROUTER_API_KEY='your-key-here'")
        return
    
    # Initialize client
    print("Initializing OpenRouter client...")
    client = Response(max_concurrent=5)
    print(f"✓ Client ready (max concurrent: {client.max_concurrent})")
    print()
    
    # ========== Demo 1: Single Generation ==========
    print("="*60)
    print("Demo 1: Single Generation")
    print("="*60)
    print()
    
    prompt = "Explain quantum entanglement in one sentence."
    print(f"Prompt: {prompt}")
    print(f"Model: anthropic/claude-3.5-sonnet")
    print("\nGenerating...")
    
    response = client.generate(
        prompt,
        model="anthropic/claude-3.5-sonnet",
        max_tokens=100,
        temperature=0.7
    )
    
    print(f"Response: {response}")
    print()
    
    # ========== Demo 2: Batch Generation ==========
    print("="*60)
    print("Demo 2: Batch Generation (5 prompts)")
    print("="*60)
    print()
    
    prompts = [
        "What is 2+2?",
        "What is the capital of France?",
        "Name one primary color.",
        "What year did humans land on the moon?",
        "What is H2O?"
    ]
    
    print(f"Generating {len(prompts)} responses concurrently...")
    print(f"Using model: openai/gpt-4o-mini")
    print()
    
    results = client.generate_batch(
        prompts,
        model="openai/gpt-4o-mini",
        max_tokens=50,
        temperature=0.5
    )
    
    print("Results:")
    print("-" * 60)
    for i, result in enumerate(results):
        if result.success:
            print(f"{i+1}. Q: {result.prompt}")
            print(f"   A: {result.response}")
            print()
        else:
            print(f"{i+1}. ERROR: {result.error}")
            print()
    
    # ========== Demo 3: Different Models ==========
    print("="*60)
    print("Demo 3: Testing Different Models")
    print("="*60)
    print()
    
    test_prompt = "Write a haiku about AI."
    models_to_test = [
        "anthropic/claude-3.5-sonnet",
        "openai/gpt-4o-mini",
        "google/gemini-2.0-flash-exp:free",
    ]
    
    print(f"Prompt: {test_prompt}")
    print(f"Testing {len(models_to_test)} different models...")
    print()
    
    for model in models_to_test:
        print(f"Model: {model}")
        try:
            response = client.generate(
                test_prompt,
                model=model,
                max_tokens=100,
                temperature=0.8
            )
            print(f"Response:\n{response}")
        except Exception as e:
            print(f"Error: {e}")
        print("-" * 60)
    
    # ========== Demo 4: Large Batch ==========
    print("="*60)
    print("Demo 4: Large Batch with Semaphore Control")
    print("="*60)
    print()
    
    # Create 20 simple math prompts
    large_batch = [f"What is {i} + {i}?" for i in range(1, 21)]
    
    print(f"Processing {len(large_batch)} prompts...")
    print(f"Max concurrent: {client.max_concurrent} (semaphore limit)")
    print()
    
    results = client.generate_batch(
        large_batch,
        model="openai/gpt-4o-mini",
        max_tokens=20,
        temperature=0.0
    )
    
    successful = sum(1 for r in results if r.success)
    print(f"✓ Completed: {successful}/{len(results)} successful")
    print()
    
    # Show first 3 results
    print("Sample results:")
    for result in results[:3]:
        if result.success:
            print(f"  Q: {result.prompt} → A: {result.response}")
    print()
    
    # ========== Summary ==========
    print("="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print()
    print("Usage:")
    print("  client = Response(max_concurrent=5)")
    print("  response = client.generate(prompt, model='...', max_tokens=100)")
    print("  results = client.generate_batch(prompts, model='...')")
    print()
    print("Features:")
    print("  ✓ Single and batch generation")
    print("  ✓ Async with semaphore control")
    print("  ✓ Model selection per request")
    print("  ✓ Configurable max_tokens, temperature")
    print("  ✓ Clean error handling")
    print()


if __name__ == "__main__":
    main()

