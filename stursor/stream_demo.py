#!/usr/bin/env python3
"""
Streaming Demo for Steered Model
Demonstrates token-by-token and interval-based streaming
"""

import torch
from model import Model


def main():
    print("="*60)
    print("STREAMING GENERATION DEMO")
    print("="*60)
    print()
    
    # Initialize model
    print("Loading model...")
    model = Model("Qwen/Qwen3-0.6B")
    print(f"✓ Model loaded: Qwen3-0.6B")
    print(f"  Hidden dim: {model.hidden_dim}, Layers: {model.num_layers}")
    print()
    
    # Test prompts
    prompts = [
        "Explain quantum computing in simple terms",
        "Write a haiku about artificial intelligence",
        "What are the three laws of robotics?"
    ]
    
    # ========== Demo 1: Token-by-token streaming ==========
    print("="*60)
    print("Demo 1: Token-by-Token Streaming (interval=1)")
    print("="*60)
    print()
    
    prompt = prompts[0]
    print(f"Prompt: {prompt}")
    print("\nGeneration (streaming):")
    print("-" * 60)
    output = model.generate(
        prompt, 
        max_new_tokens=50, 
        stream=True,
        stream_interval=1  # Every token
    )
    print("-" * 60)
    print()
    
    # ========== Demo 2: Every 5 tokens ==========
    print("="*60)
    print("Demo 2: Stream Every 5 Tokens (interval=5)")
    print("="*60)
    print()
    
    prompt = prompts[1]
    print(f"Prompt: {prompt}")
    print("\nGeneration (streaming every 5 tokens):")
    print("-" * 60)
    output = model.generate(
        prompt,
        max_new_tokens=40,
        stream=True,
        stream_interval=5  # Every 5 tokens
    )
    print("-" * 60)
    print()
    
    # ========== Demo 3: Streaming with steering ==========
    print("="*60)
    print("Demo 3: Streaming with Steering Active")
    print("="*60)
    print()
    
    # Add random steering vector to middle layer
    layer_idx = model.num_layers // 2
    layer_name = f"model.layers.{layer_idx}"
    steering_vector = torch.randn(1, 1, model.hidden_dim) * 0.5
    
    print(f"Adding steering to {layer_name} (scale=1.5)")
    model.add_steering_vector(layer_name, steering_vector, scale=1.5)
    print(f"✓ Active steerings: {list(model.get_active_steerings().keys())}")
    print()
    
    prompt = prompts[2]
    print(f"Prompt: {prompt}")
    print("\nGeneration with steering (streaming):")
    print("-" * 60)
    output = model.generate(
        prompt,
        max_new_tokens=50,
        stream=True,
        stream_interval=1
    )
    print("-" * 60)
    print()
    
    # Clear steering
    model.clear_all_steering()
    
    # ========== Demo 4: Multiple prompts with streaming ==========
    print("="*60)
    print("Demo 4: Batch Streaming (Multiple Prompts)")
    print("="*60)
    print()
    
    batch_prompts = [
        "Count from 1 to 5",
        "Name three colors"
    ]
    
    print("Generating for multiple prompts with streaming...")
    print()
    outputs = model.generate(
        batch_prompts,
        max_new_tokens=30,
        stream=True,
        stream_interval=3  # Every 3 tokens
    )
    print()
    
    # ========== Summary ==========
    print("="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print()
    print("Streaming options:")
    print("  - stream=True: Enable streaming output")
    print("  - stream_interval=1: Output every token (default)")
    print("  - stream_interval=N: Output every N tokens")
    print()
    print("Streaming works with:")
    print("  ✓ Single prompts")
    print("  ✓ Multiple prompts (batch)")
    print("  ✓ Steering vectors active")
    print("  ✓ All hook management features")
    print()


if __name__ == "__main__":
    main()

