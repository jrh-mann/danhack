"""
Steered Model Class for nnsight with PyTorch Hooks
Designed for Qwen3 architecture (compatible with similar LLaMA-style models)
"""

import torch
from nnsight import LanguageModel
import matplotlib.pyplot as plt
import openai
from typing import Union, List, Dict, Tuple, Optional, Callable
from transformers import TextStreamer
import sys


class IntervalStreamer(TextStreamer):
    """Custom TextStreamer that outputs every N tokens instead of every token"""
    
    def __init__(self, tokenizer, interval: int = 1, skip_prompt: bool = True, **kwargs):
        super().__init__(tokenizer, skip_prompt=skip_prompt, **kwargs)
        self.interval = interval
        self.token_count = 0
        self.buffer = []
        
    def on_finalized_text(self, text: str, stream_end: bool = False):
        """Called when text is finalized (after decode)"""
        self.buffer.append(text)
        self.token_count += 1
        
        # Output every N tokens or at the end
        if self.token_count % self.interval == 0 or stream_end:
            output = ''.join(self.buffer)
            print(output, end='', flush=True)
            self.buffer = []
            
        # Print newline at the very end
        if stream_end:
            print()


class Model:
    def __init__(self, model_name, device=None):
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # Load model with explicit dispatch parameter to avoid meta tensors
        self.model = LanguageModel(
            model_name,
            dispatch=True,  # This ensures model is actually loaded to device
            device_map={"": device},  # Map all layers to single device
        )
        
        # Store active hook handles
        self.hook_handles: Dict[str, torch.utils.hooks.RemovableHandle] = {}
        # Store steering vectors: {layer_name: (vector, scale)}
        self.steering_vectors: Dict[str, Tuple[torch.Tensor, float]] = {}
        
        # Cache architecture info
        self._num_layers = None
        self._hidden_dim = None
    
    @property
    def num_layers(self) -> int:
        """Get number of transformer layers in the model"""
        if self._num_layers is None:
            self._num_layers = len(self.model._model.model.layers)
        return self._num_layers
    
    @property
    def hidden_dim(self) -> int:
        """Get hidden dimension size of the model"""
        if self._hidden_dim is None:
            self._hidden_dim = self.model._model.model.embed_tokens.weight.shape[1]
        return self._hidden_dim
        
    def _get_module(self, layer_name: str):
        """Get a module from the model by name (e.g., 'model.layers.10')"""
        # For Qwen3, the structure is: model._model.model.layers[i]
        module = self.model._model  # Get the full model (with LM head)
        for part in layer_name.split('.'):
            module = getattr(module, part)
        return module
    
    def _create_steering_hook(self, layer_name: str):
        """Create a hook function that adds the steering vector"""
        def hook_fn(module, input, output):
            if layer_name not in self.steering_vectors:
                return output
            
            steering_vector, scale = self.steering_vectors[layer_name]
            
            # Handle different output types (tuple, tensor, etc.)
            if isinstance(output, tuple):
                # For transformer layers that return (hidden_states, ...)
                hidden_states = output[0]
                steered = hidden_states + scale * steering_vector.to(hidden_states.device)
                return (steered,) + output[1:]
            else:
                # Direct tensor output
                return output + scale * steering_vector.to(output.device)
        
        return hook_fn
    
    def add_steering_vector(self, layer_name: str, vector: torch.Tensor, scale: float = 1.0):
        """
        Add a steering vector to a specific layer.
        
        Args:
            layer_name: Name of the layer to steer (e.g., 'layers.10.mlp')
            vector: Steering vector to add (shape should match layer output)
            scale: Scaling factor for the steering vector
        """
        # If already exists, remove first
        if layer_name in self.hook_handles:
            self.remove_steering_vector(layer_name)
        
        # Store the steering vector
        self.steering_vectors[layer_name] = (vector, scale)
        
        # Get the target module and register hook
        module = self._get_module(layer_name)
        hook_fn = self._create_steering_hook(layer_name)
        handle = module.register_forward_hook(hook_fn)
        
        # Store the hook handle
        self.hook_handles[layer_name] = handle
        
    def remove_steering_vector(self, layer_name: str):
        """
        Remove steering vector and hook from a specific layer.
        
        Args:
            layer_name: Name of the layer to remove steering from
        """
        if layer_name in self.hook_handles:
            # Remove the hook
            self.hook_handles[layer_name].remove()
            del self.hook_handles[layer_name]
        
        if layer_name in self.steering_vectors:
            del self.steering_vectors[layer_name]
    
    def update_steering_vector(self, layer_name: str, vector: torch.Tensor, scale: float = 1.0):
        """
        Update an existing steering vector (removes old one and adds new one).
        
        Args:
            layer_name: Name of the layer to update
            vector: New steering vector
            scale: New scaling factor
        """
        self.remove_steering_vector(layer_name)
        self.add_steering_vector(layer_name, vector, scale)
    
    def clear_all_steering(self):
        """Remove all steering vectors and hooks"""
        layer_names = list(self.hook_handles.keys())
        for layer_name in layer_names:
            self.remove_steering_vector(layer_name)
    
    def get_active_steerings(self) -> Dict[str, Tuple[torch.Tensor, float]]:
        """Return a copy of currently active steering vectors"""
        return self.steering_vectors.copy()

    def generate(
        self, 
        prompts: Union[str, List[str]], 
        max_new_tokens: int = 100, 
        stream: bool = False,
        stream_interval: int = 1,
        **kwargs
    ):
        """
        Generate text with steering applied.
        
        Args:
            prompts: Single prompt string or list of prompts
            max_new_tokens: Maximum number of tokens to generate
            stream: If True, print tokens as they're generated
            stream_interval: Print every N tokens (default: 1 for token-by-token)
            **kwargs: Additional arguments to pass to model.generate()
            
        Returns:
            Generated text (single string if input is string, list if input is list)
        """
        # Normalize input to list
        is_single = isinstance(prompts, str)
        prompt_list = [prompts] if is_single else prompts

        prompt_list = [self.model.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=False,
            add_generation_prompt=True
            ) for prompt in prompt_list]
        
        # Generate with hooks active
        outputs = []
        for i, prompt in enumerate(prompt_list):
            # Create streamer if streaming is enabled
            if stream:
                if len(prompt_list) > 1:
                    print(f"\n[Prompt {i+1}/{len(prompt_list)}]")
                streamer = IntervalStreamer(
                    self.model.tokenizer, 
                    interval=stream_interval,
                    skip_prompt=True
                )
            else:
                streamer = None
            
            # Tokenize input and move to device
            inputs = self.model.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate tokens (use _model which is the full model with LM head)
            output_ids = self.model._model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                streamer=streamer,
                **kwargs
            )
            
            # Decode output
            output_text = self.model.tokenizer.decode(output_ids[0], skip_special_tokens=True)
            outputs.append(output_text)
        
        # Return single string or list based on input
        return outputs[0] if is_single else outputs

    def get_activations(self, prompts, layer_names=[10, 15, 20]):
        """
        Get activations from specified layers for given prompts.
        For Qwen3 architecture: uses model._model.model.layers[i]
        
        Args:
            prompts: Single prompt or list of prompts
            layer_names: List of layer indices to extract activations from
            
        Returns:
            List of saved activation tensors
        """
        activations = []

        with torch.no_grad():
            with self.model.trace(prompts) as trace:
                # Qwen3 architecture: model.layers[i]
                for layer in layer_names:
                    activations.append(self.model._model.model.layers[layer].output.save())
                
        return activations