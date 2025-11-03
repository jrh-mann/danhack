#!/usr/bin/env python3
"""
Steered Model Testing Script
Testing all functionality of the steered nnsight model with PyTorch hooks
"""

import torch
import numpy as np
from model import Model
import matplotlib.pyplot as plt


def main():
    print("="*60)
    print("STEERED MODEL TESTING")
    print("="*60)
    print()
    
    # ==================== 1. Initialize Model ====================
    print("1. Initializing Model...")
    model_name = "Qwen/Qwen3-0.6B"  # or "EleutherAI/pythia-70m" or "microsoft/phi-2"
    model = Model(model_name)
    print(f"✓ Loaded model: {model_name}")
    print()
    
    # ==================== 2. Baseline Generation ====================
    print("2. Testing Baseline Generation (No Steering)...")
    
    # Test single prompt
    prompt = "The weather today is"
    baseline_output = model.generate(prompt, max_new_tokens=30)
    print("✓ Baseline Output (single prompt):")
    print(f"  {baseline_output}")
    print()
    
    # Test multiple prompts
    prompts = [
        "The capital of France is",
        "Artificial intelligence is",
        "The meaning of life is"
    ]
    baseline_outputs = model.generate(prompts, max_new_tokens=20)
    print("✓ Baseline Outputs (multiple prompts):")
    for i, output in enumerate(baseline_outputs):
        print(f"  {i+1}. {output}")
    print()
    
    # ==================== 3. Get Model Architecture Info ====================
    print("3. Inspecting Model Architecture...")
    
    print(f"Model: {model_name}")
    print(f"Architecture: Qwen3 ({type(model.model._model).__name__})")
    print(f"✓ Hidden dimension: {model.hidden_dim}")
    print(f"✓ Number of layers: {model.num_layers}")
    
    # Qwen3 uses "model.layers.X" format
    layer_prefix = "model.layers"
    hidden_dim = model.hidden_dim
    num_layers = model.num_layers
    print()
    
    # ==================== 4. Create Random Steering Vectors ====================
    print("4. Creating Random Steering Vectors...")
    
    # Shape: (1, 1, hidden_dim) for broadcasting across batch and sequence
    random_vector_1 = torch.randn(1, 1, hidden_dim) * 0.1  # Small magnitude
    random_vector_2 = torch.randn(1, 1, hidden_dim) * 0.5  # Medium magnitude
    random_vector_3 = torch.randn(1, 1, hidden_dim) * 1.0  # Larger magnitude
    
    print(f"✓ Created random steering vectors with shapes: {random_vector_1.shape}")
    print()
    
    # ==================== 5. Test Adding Steering Vectors ====================
    print("5. Testing Adding Steering Vectors...")
    
    target_layer = f"{layer_prefix}.{num_layers // 2}"
    print(f"Adding steering to layer: {target_layer}")
    
    model.add_steering_vector(target_layer, random_vector_1, scale=1.0)
    
    # Check active steerings
    active = model.get_active_steerings()
    print(f"✓ Active steerings: {list(active.keys())}")
    
    # Generate with steering
    steered_output_1 = model.generate(prompt, max_new_tokens=30)
    print("✓ Steered Output (small random noise):")
    print(f"  {steered_output_1}")
    print("\nComparison:")
    print(f"  Baseline: {baseline_output}")
    print(f"  Steered:  {steered_output_1}")
    print()
    
    # ==================== 6. Test Updating Steering Vectors ====================
    print("6. Testing Updating Steering Vectors...")
    
    print(f"Updating steering vector for layer: {target_layer}")
    model.update_steering_vector(target_layer, random_vector_2, scale=2.0)
    
    steered_output_2 = model.generate(prompt, max_new_tokens=30)
    print("✓ Steered Output (medium random noise, scale=2.0):")
    print(f"  {steered_output_2}")
    print("\nComparison:")
    print(f"  Baseline:       {baseline_output}")
    print(f"  Small steering: {steered_output_1}")
    print(f"  Med steering:   {steered_output_2}")
    print()
    
    # ==================== 7. Test Multiple Steering Vectors ====================
    print("7. Testing Multiple Steering Vectors...")
    
    # Clear existing steering
    model.clear_all_steering()
    print("✓ Cleared all steering")
    print(f"  Active steerings: {list(model.get_active_steerings().keys())}")
    
    # Add steering to multiple layers
    layers_to_steer = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
    print(f"\nAdding steering to layers: {layers_to_steer}")
    
    for i, layer_idx in enumerate(layers_to_steer):
        layer_name = f"{layer_prefix}.{layer_idx}"
        vector = torch.randn(1, 1, hidden_dim) * 0.3
        model.add_steering_vector(layer_name, vector, scale=1.0)
        print(f"  ✓ Added steering to {layer_name}")
    
    print(f"\n✓ Active steerings: {list(model.get_active_steerings().keys())}")
    
    # Generate with multiple steerings
    multi_steered_output = model.generate(prompt, max_new_tokens=30)
    print("\n✓ Multi-layer Steered Output:")
    print(f"  {multi_steered_output}")
    print("\nComparison:")
    print(f"  Baseline:     {baseline_output}")
    print(f"  Multi-steer:  {multi_steered_output}")
    print()
    
    # ==================== 8. Test with Multiple Prompts and Steering ====================
    print("8. Testing Multiple Prompts with Steering Active...")
    
    test_prompts = [
        "Once upon a time",
        "The quick brown fox",
        "In the beginning"
    ]
    
    steered_multi_outputs = model.generate(test_prompts, max_new_tokens=20)
    
    print("✓ Multiple Prompts with Multi-layer Steering:")
    for i, output in enumerate(steered_multi_outputs):
        print(f"  {i+1}. {output}")
    print()
    
    # ==================== 9. Test Removing Individual Steering Vectors ====================
    print("9. Testing Removing Individual Steering Vectors...")
    
    layer_to_remove = f"{layer_prefix}.{layers_to_steer[0]}"
    print(f"Removing steering from: {layer_to_remove}")
    model.remove_steering_vector(layer_to_remove)
    
    print(f"✓ Active steerings after removal: {list(model.get_active_steerings().keys())}")
    
    # Generate again
    output_after_removal = model.generate(prompt, max_new_tokens=30)
    print("\n✓ Output after removing one steering:")
    print(f"  {output_after_removal}")
    print()
    
    # ==================== 10. Test Different Steering Scales ====================
    print("10. Testing Different Steering Scales...")
    
    # Clear and test different scales
    model.clear_all_steering()
    
    test_layer = f"{layer_prefix}.{num_layers // 2}"
    test_vector = torch.randn(1, 1, hidden_dim) * 0.5
    
    scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    test_prompt = "The future of technology is"
    
    print(f"Testing different scales on layer {test_layer}:")
    print(f"Prompt: {test_prompt}\n")
    
    for scale in scales:
        model.update_steering_vector(test_layer, test_vector, scale=scale)
        output = model.generate(test_prompt, max_new_tokens=25)
        print(f"  Scale {scale:.1f}: {output}")
    print()
    
    # ==================== 11. Test Extreme Steering ====================
    print("11. Testing Extreme Steering (Chaotic Behavior)...")
    
    # Test with very large random noise
    model.clear_all_steering()
    
    large_random = torch.randn(1, 1, hidden_dim) * 5.0
    model.add_steering_vector(test_layer, large_random, scale=10.0)
    
    print("Testing extreme steering (large random noise, scale=10.0):")
    extreme_output = model.generate("Hello world", max_new_tokens=30)
    print(f"✓ Output: {extreme_output}")
    print("  (Note: This may produce gibberish or unexpected tokens)")
    print()
    
    # ==================== 12. Test Clearing All Steering ====================
    print("12. Testing Clearing All Steering...")
    
    # Clear all steering
    model.clear_all_steering()
    print("✓ All steering cleared")
    print(f"  Active steerings: {list(model.get_active_steerings().keys())}")
    
    # Generate should be back to baseline
    final_output = model.generate(prompt, max_new_tokens=30)
    print("\n✓ Output after clearing all steering:")
    print(f"  {final_output}")
    print("\nOriginal baseline:")
    print(f"  {baseline_output}")
    print("  (Should be very similar or identical)")
    print()
    
    # ==================== 13. Visualize Steering Vector Statistics ====================
    print("13. Visualizing Steering Vector Statistics...")
    
    # Create and visualize different steering vectors
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    vectors = [
        (torch.randn(hidden_dim) * 0.1, "Small (std=0.1)"),
        (torch.randn(hidden_dim) * 0.5, "Medium (std=0.5)"),
        (torch.randn(hidden_dim) * 1.0, "Large (std=1.0)"),
        (torch.randn(hidden_dim) * 2.0, "Very Large (std=2.0)")
    ]
    
    for ax, (vec, title) in zip(axes.flat, vectors):
        ax.hist(vec.numpy(), bins=50, alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel("Value")
        ax.set_ylabel("Frequency")
        ax.axvline(0, color='red', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('steering_vectors_distribution.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to steering_vectors_distribution.png")
    print()
    
    # ==================== 14. Summary ====================
    print("="*60)
    print("TESTING SUMMARY")
    print("="*60)
    print("✓ Baseline generation (single and multiple prompts)")
    print("✓ Adding steering vectors to single layer")
    print("✓ Updating steering vectors (remove + add)")
    print("✓ Multiple steering vectors on different layers")
    print("✓ Removing individual steering vectors")
    print("✓ Different steering scales")
    print("✓ Extreme/chaotic steering with large noise")
    print("✓ Clearing all steering")
    print("✓ Batch generation with steering active")
    print("✓ Visualization of steering vector distributions")
    print("="*60)
    print("All tests completed successfully!")
    print()


if __name__ == "__main__":
    main()

