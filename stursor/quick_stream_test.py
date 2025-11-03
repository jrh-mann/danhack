#!/usr/bin/env python3
"""Quick test of streaming functionality"""

from model import Model

# Load model
print("Loading model...")
model = Model("Qwen/Qwen3-0.6B")
print("Model loaded!\n")

# Test 1: No streaming
print("=" * 60)
print("Test 1: Normal generation (no streaming)")
print("=" * 60)
output = model.generate("Write a short joke", max_new_tokens=30, stream=False)
print(output)
print()

# Test 2: Token-by-token streaming
print("=" * 60)
print("Test 2: Token-by-token streaming")
print("=" * 60)
output = model.generate("Write a short joke", max_new_tokens=30, stream=True, stream_interval=1)
print()

# Test 3: Every 5 tokens
print("=" * 60)
print("Test 3: Stream every 5 tokens")
print("=" * 60)
output = model.generate("Tell me about space", max_new_tokens=40, stream=True, stream_interval=5)
print()

print("✓ All streaming tests complete!")

