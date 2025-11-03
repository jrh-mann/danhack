"""
Create steering vectors from examples using contrastive activation differences
"""

import torch
from typing import List, Optional, Tuple
from model import Model
from response import Response


def create_direction(
    concept: str,
    model: Model,
    api_client: Response,
    layer: int = 14,
    layers: Optional[List[int]] = None,
    n_examples: int = 10,
    api_model: str = "openai/gpt-4o-mini",
    max_tokens: int = 100,
    temperature: float = 0.8,
    verbose: bool = True
) -> Tuple[torch.Tensor, dict]:
    """
    Create a steering vector by generating contrastive examples and computing activation differences.
    
    Process:
    1. Use API to generate positive examples (concept present)
    2. Use API to generate negative examples (concept absent)
    3. Get activations from local model for each example
    4. Compute mean difference: mean(positive) - mean(negative)
    
    Args:
        concept: Description of the behavior to steer (e.g., "sarcastic responses")
        model: Local Model instance for getting activations
        api_client: Response instance for generating examples
        layer: Single layer index (used if layers is None)
        layers: List of layer indices to create steering vectors for (overrides layer)
        n_examples: Number of positive/negative examples to generate
        api_model: OpenRouter model to use for generation
        max_tokens: Max tokens per generated example
        temperature: Temperature for generation
        verbose: Print progress
        
    Returns:
        Tuple of (steering_vectors, metadata_dict)
        - steering_vectors: Dict[int, torch.Tensor] mapping layer -> normalized vector (1, 1, hidden_dim)
                           OR single torch.Tensor if only one layer
        - metadata: dict with generation info
    """
    
    # Handle multi-layer or single-layer
    if layers is not None:
        target_layers = layers
    else:
        target_layers = [layer]
    
    if verbose:
        print(f"Creating steering direction for: '{concept}'")
        print(f"Layers: {target_layers}, Examples: {n_examples}")
        print()
    
    # ========== Step 1: Generate both positive and negative examples in one call ==========
    if verbose:
        print("Step 1: Generating positive and negative examples in one API call...")
    
    combined_prompt = f"""I am running a steering vector experiment to teach a model about the concept: "{concept}".

Your task: Generate {n_examples} pairs of contrasting text examples.

For each pair, provide:
1. A POSITIVE example that clearly demonstrates {concept}
2. A NEGATIVE example that does NOT demonstrate {concept} (should be neutral or opposite)

Each example should be 1-2 sentences long and representative.

Use this EXACT format:

POSITIVE:
[example 1]
[example 2]
[example 3]
...

NEGATIVE:
[example 1]
[example 2]
[example 3]
...

Generate {n_examples} high-quality contrasting examples now:"""
    
    combined_response = api_client.generate(
        combined_prompt,
        model=api_model,
        max_tokens=max_tokens * n_examples * 2,  # Need space for both sets
        temperature=temperature
    )
    
    if verbose:
        print(f"✓ Received API response, parsing examples...")
    
    # Parse using regex to extract POSITIVE and NEGATIVE sections
    import re
    
    # Extract positive examples
    positive_match = re.search(r'POSITIVE:\s*\n(.*?)(?=NEGATIVE:|$)', combined_response, re.DOTALL | re.IGNORECASE)
    if positive_match:
        positive_text = positive_match.group(1)
        positive_examples = [
            line.strip().lstrip('0123456789.-) ')
            for line in positive_text.split('\n')
            if line.strip() and len(line.strip()) > 10
        ][:n_examples]
    else:
        raise ValueError(f"Could not parse POSITIVE examples from API response. Response:\n{combined_response[:500]}")
    
    # Extract negative examples
    negative_match = re.search(r'NEGATIVE:\s*\n(.*?)$', combined_response, re.DOTALL | re.IGNORECASE)
    if negative_match:
        negative_text = negative_match.group(1)
        negative_examples = [
            line.strip().lstrip('0123456789.-) ')
            for line in negative_text.split('\n')
            if line.strip() and len(line.strip()) > 10
        ][:n_examples]
    else:
        raise ValueError(f"Could not parse NEGATIVE examples from API response. Response:\n{combined_response[:500]}")
    
    if verbose:
        print(f"✓ Parsed {len(positive_examples)} positive and {len(negative_examples)} negative examples")
        print(f"\nPositive examples (demonstrating {concept}):")
        for i, ex in enumerate(positive_examples[:3], 1):
            print(f"  {i}. {ex[:80]}...")
        print(f"\nNegative examples (NOT demonstrating {concept}):")
        for i, ex in enumerate(negative_examples[:3], 1):
            print(f"  {i}. {ex[:80]}...")
        print()
    
    # ========== Step 2: Get activations for positive examples (all layers at once) ==========
    if verbose:
        print(f"Step 2: Getting activations for positive examples across {len(target_layers)} layers...")
    
    # Store activations per layer: {layer_idx: [activation1, activation2, ...]}
    positive_activations_by_layer = {layer_idx: [] for layer_idx in target_layers}
    
    for example in positive_examples:
        # Tokenize the example
        inputs = model.model.tokenizer(example, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            # Run forward pass once and extract activations from all layers
            outputs = model.model._model(**inputs, output_hidden_states=True)
            
            # Extract activations for each target layer
            for layer_idx in target_layers:
                # output_hidden_states returns tuple: (embeddings, layer1, layer2, ..., layerN)
                # So layer N is at index N+1 (because index 0 is embeddings)
                hidden_states = outputs.hidden_states[layer_idx + 1]  # Shape: [batch, seq, hidden]
                
                # Mean across sequence dimension
                mean_act = hidden_states.mean(dim=1).squeeze(0)  # Shape: [hidden_dim]
                positive_activations_by_layer[layer_idx].append(mean_act.cpu())
    
    # Compute mean for each layer
    positive_means = {}
    for layer_idx in target_layers:
        positive_means[layer_idx] = torch.stack(positive_activations_by_layer[layer_idx]).mean(dim=0)
    
    if verbose:
        print(f"✓ Computed positive activations for {len(target_layers)} layers")
        print()
    
    # ========== Step 3: Get activations for negative examples (all layers at once) ==========
    if verbose:
        print(f"Step 3: Getting activations for negative examples across {len(target_layers)} layers...")
    
    negative_activations_by_layer = {layer_idx: [] for layer_idx in target_layers}
    
    for example in negative_examples:
        # Tokenize the example
        inputs = model.model.tokenizer(example, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            # Run forward pass once and extract activations from all layers
            outputs = model.model._model(**inputs, output_hidden_states=True)
            
            # Extract activations for each target layer
            for layer_idx in target_layers:
                hidden_states = outputs.hidden_states[layer_idx + 1]  # Shape: [batch, seq, hidden]
                mean_act = hidden_states.mean(dim=1).squeeze(0)  # Shape: [hidden_dim]
                negative_activations_by_layer[layer_idx].append(mean_act.cpu())
    
    # Compute mean for each layer
    negative_means = {}
    for layer_idx in target_layers:
        negative_means[layer_idx] = torch.stack(negative_activations_by_layer[layer_idx]).mean(dim=0)
    
    if verbose:
        print(f"✓ Computed negative activations for {len(target_layers)} layers")
        print()
    
    # ========== Step 4: Compute steering vectors (one per layer, all normalized) ==========
    if verbose:
        print(f"Step 4: Computing steering vectors for {len(target_layers)} layers...")
    
    steering_vectors = {}
    original_norms = {}
    
    for layer_idx in target_layers:
        # Difference: positive - negative
        steering_vector = positive_means[layer_idx] - negative_means[layer_idx]
        
        # Store original norm
        original_norms[layer_idx] = steering_vector.norm().item()
        
        # Normalize the vector (L2 normalization)
        steering_vector = steering_vector / steering_vector.norm()
        
        # Reshape for model compatibility: (1, 1, hidden_dim)
        steering_vector = steering_vector.unsqueeze(0).unsqueeze(0)
        
        steering_vectors[layer_idx] = steering_vector
    
    if verbose:
        print(f"✓ Created {len(steering_vectors)} normalized steering vectors")
        for layer_idx in target_layers:
            print(f"  Layer {layer_idx}: norm={original_norms[layer_idx]:.4f} → 1.0")
        print()
    
    # Metadata
    metadata = {
        "concept": concept,
        "layers": target_layers,
        "layer": target_layers[0] if len(target_layers) == 1 else None,  # For backward compatibility
        "n_positive_examples": len(positive_examples),
        "n_negative_examples": len(negative_examples),
        "positive_examples": positive_examples,
        "negative_examples": negative_examples,
        "original_norms": original_norms,
        "hidden_dim": list(steering_vectors.values())[0].shape[-1]
    }
    
    # Return single vector if only one layer (backward compatibility)
    if len(target_layers) == 1:
        return steering_vectors[target_layers[0]], metadata
    
    return steering_vectors, metadata


def create_direction_multi_layer(
    concept: str,
    model: Model,
    api_client: Response,
    layers: List[int] = [10, 14, 18, 22],
    n_examples: int = 10,
    api_model: str = "openai/gpt-4o-mini",
    max_tokens: int = 100,
    temperature: float = 0.8,
    verbose: bool = True
) -> dict:
    """
    Create steering vectors for multiple layers at once.
    
    Args:
        concept: Description of the behavior to steer
        model: Local Model instance
        api_client: Response instance
        layers: List of layer indices to extract vectors for
        n_examples: Number of positive/negative examples
        api_model: OpenRouter model to use
        max_tokens: Max tokens per example
        temperature: Temperature for generation
        verbose: Print progress
        
    Returns:
        Dictionary mapping layer_idx -> (steering_vector, metadata)
    """
    
    if verbose:
        print(f"Creating multi-layer steering directions for: '{concept}'")
        print(f"Layers: {layers}, Examples per layer: {n_examples}")
        print("="*60)
        print()
    
    results = {}
    
    for layer_idx in layers:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Processing Layer {layer_idx}")
            print(f"{'='*60}\n")
        
        vector, metadata = create_direction(
            concept=concept,
            model=model,
            api_client=api_client,
            layer=layer_idx,
            n_examples=n_examples,
            api_model=api_model,
            max_tokens=max_tokens,
            temperature=temperature,
            verbose=verbose
        )
        
        results[layer_idx] = (vector, metadata)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"✓ Created steering vectors for {len(results)} layers")
        print(f"{'='*60}\n")
    
    return results


# Example usage
if __name__ == "__main__":
    import os
    
    print("="*60)
    print("STEERING DIRECTION CREATION DEMO")
    print("="*60)
    print()
    
    # Initialize
    print("Loading models...")
    local_model = Model("Qwen/Qwen3-0.6B")
    api_client = Response(api_key=os.getenv("OPENROUTER_API_KEY"))
    print("✓ Models loaded\n")
    
    # Create a steering direction
    concept = "sarcastic and witty responses"
    
    vector, metadata = create_direction(
        concept=concept,
        model=local_model,
        api_client=api_client,
        layer=14,
        n_examples=5,
        api_model="openai/gpt-4o-mini"
    )
    
    print("="*60)
    print("RESULTS")
    print("="*60)
    print(f"Concept: {metadata['concept']}")
    print(f"Layer: {metadata['layer']}")
    print(f"Vector shape: {vector.shape}")
    print(f"Vector norm: {metadata['vector_norm']:.4f}")
    print()
    
    print("Testing steering vector...")
    # Apply steering
    local_model.add_steering_vector(
        f"model.layers.{metadata['layer']}",
        vector,
        scale=2.0
    )
    
    test_prompt = "What do you think about Mondays?"
    print(f"\nTest prompt: {test_prompt}")
    print("\nGeneration with steering (scale=2.0):")
    print("-"*60)
    response = local_model.generate(test_prompt, max_new_tokens=50, stream=True)
    print("-"*60)
