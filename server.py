#!/root/danhack/.venv/bin/python3
"""
WebSocket Server for Steering Vector Chatbot
Single-file server connecting React frontend to Python steering backend
"""

import os
import json
import asyncio
import uuid
import re
from typing import Dict, Any
from aiohttp import web
import torch

# Import your existing modules
import sys
sys.path.append('./stursor')
from model import Model
from response import Response
from create_direction import create_direction


class SteeringServer:
    """Manages model state and steering vectors"""
    
    def __init__(self):
        self.model = None
        self.api_client = None
        self.steering_vectors: Dict[str, Dict[str, Any]] = {}
        # Format: {slider_id: {"vector": tensor, "layer": int, "label": str}}
        self.conversation_history: Dict[str, list] = {}
        # Format: {ws_id: [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
        
    async def initialize(self):
        """Load model on startup"""
        print("🔄 Loading model...")
        if not os.getenv("OPENROUTER_API_KEY"):
            raise ValueError("Set OPENROUTER_API_KEY environment variable")
        
        # Load local model (this may take a while)
        self.model = Model("Qwen/Qwen3-4B")
        self.api_client = Response(max_concurrent=3)
        print("✅ Model loaded and ready!")
        
    async def create_steering_from_idea(self, idea: str) -> Dict[str, Any]:
        """Create steering vectors from user idea (multi-layer), return slider spec"""
        slider_id = str(uuid.uuid4())
        # Use multiple layers for stronger, more robust steering
        layers = [10, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]  # Spread across middle-to-upper layers
        
        print(f"🎯 Creating steering vectors for: '{idea}' (layers: {layers})")
        
        # Create steering vectors for all layers (runs in executor to not block)
        loop = asyncio.get_event_loop()
        vectors_or_dict, metadata = await loop.run_in_executor(
            None,
            lambda: create_direction(
                concept=idea,
                model=self.model,
                api_client=self.api_client,
                layers=layers,  # Multi-layer
                n_examples=8,  # Fewer examples for faster generation
                api_model="qwen/qwen3-235b-a22b-2507",
                verbose=True
            )
        )
        
        # Store steering vectors (dict mapping layer -> vector)
        self.steering_vectors[slider_id] = {
            "vectors": vectors_or_dict,  # Dict[int, torch.Tensor]
            "layers": metadata["layers"],
            "label": idea[:50]  # Truncate long labels
        }
        
        print(f"✅ Steering vectors created for {len(layers)} layers: {slider_id}")
        
        # Return slider specification for frontend
        return {
            "id": slider_id,
            "label": idea[:50],
            "min": -25.0,  # Slider range (for UI convenience)
            "max": 25.0,
            "step": 0.5,
            "value": 0.0  # Start at zero (neutral)
        }
    
    async def generate_with_steering(self, message: str, slider_states: list, ws: web.WebSocketResponse, ws_id: str):
        """Generate response with active steering vectors, stream to websocket"""
        
        # Add user message to history
        if ws_id not in self.conversation_history:
            self.conversation_history[ws_id] = []
        
        self.conversation_history[ws_id].append({
            "role": "user",
            "content": message
        })
        
        # Apply all steering vectors at their current slider values
        self.model.clear_all_steering()
        
        for slider in slider_states:
            slider_id = slider["id"]
            scale = slider["value"]
            
            if slider_id in self.steering_vectors and abs(scale) > 0.01:
                vec_data = self.steering_vectors[slider_id]
                vectors = vec_data["vectors"]  # Dict[int, torch.Tensor]
                layers = vec_data["layers"]
                
                # Apply same scale to all layers
                for layer_idx, vector in vectors.items():
                    layer_name = f"model.layers.{layer_idx}"
                    self.model.add_steering_vector(layer_name, vector, scale=scale)
                
                print(f"📊 Applied steering '{vec_data['label']}' at scale {scale:.2f} across {len(layers)} layers: {layers}")
        
        # Generate response (run in executor to avoid blocking event loop)
        loop = asyncio.get_event_loop()
        
        def generate_sync():
            # Format prompt with chat template (disable thinking) - use full conversation history
            prompt_formatted = self.model.model.tokenizer.apply_chat_template(
                self.conversation_history[ws_id],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
            
            # Tokenize
            inputs = self.model.model.tokenizer(prompt_formatted, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            input_length = inputs["input_ids"].shape[1]
            
            # Generate
            output_ids = self.model.model._model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9
            )
            
            # Decode only the new tokens (skip the prompt)
            new_tokens = output_ids[0][input_length:]
            response = self.model.model.tokenizer.decode(new_tokens, skip_special_tokens=True)
            return response
        
        assistant_response = await loop.run_in_executor(None, generate_sync)
        assistant_response = assistant_response.strip()
        
        # Remove any think tags that might appear (even though thinking is disabled)
        assistant_response = re.sub(r'<think>\s*</think>', '', assistant_response)
        assistant_response = re.sub(r'<think>.*?</think>', '', assistant_response, flags=re.DOTALL)
        assistant_response = assistant_response.strip()
        
        # Add assistant response to history
        self.conversation_history[ws_id].append({
            "role": "assistant",
            "content": assistant_response
        })
        
        print(f"🤖 Response ({len(assistant_response)} chars): {assistant_response[:100]}...")
        print(f"📚 History length: {len(self.conversation_history[ws_id])} messages")
        
        # Stream the response word by word for smooth UI
        if assistant_response:
            words = assistant_response.split()
            for i, word in enumerate(words):
                token = word + " " if i < len(words) - 1 else word
                await ws.send_json({
                    "type": "assistant_token",
                    "token": token
                })
                await asyncio.sleep(0.05)  # Smooth streaming effect
        
        # Send done signal
        await ws.send_json({"type": "assistant_done"})
        print("✅ Generation complete")


# Global server instance
server = SteeringServer()


async def websocket_handler(request):
    """Handle WebSocket connections"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    # Generate unique ID for this connection
    ws_id = str(uuid.uuid4())
    
    print(f"🔌 Client connected (ID: {ws_id[:8]}...)")
    
    # Send welcome message
    await ws.send_json({
        "type": "system",
        "content": "Connected! Submit a steering idea to get started."
    })
    
    try:
        async for msg in ws:
            if msg.type == web.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                    msg_type = data.get("type")
                    
                    if msg_type == "idea":
                        # Create steering vector from idea
                        idea = data.get("idea", "").strip()
                        if not idea:
                            continue
                            
                        await ws.send_json({
                            "type": "system",
                            "content": f"Creating steering vector for: '{idea}'. This may take 30-60 seconds..."
                        })
                        
                        try:
                            slider_spec = await server.create_steering_from_idea(idea)
                            await ws.send_json({
                                "type": "add_slider",
                                "slider": slider_spec
                            })
                            await ws.send_json({
                                "type": "system",
                                "content": f"✅ Steering vector created! Adjust the slider to control intensity."
                            })
                        except Exception as e:
                            print(f"❌ Error creating steering: {e}")
                            await ws.send_json({
                                "type": "system",
                                "content": f"❌ Error creating steering: {str(e)}"
                            })
                    
                    elif msg_type == "chat":
                        # Generate response with steering
                        message = data.get("message", "").strip()
                        if not message:
                            continue
                        
                        state = data.get("state", {})
                        sliders = state.get("sliders", [])
                        
                        try:
                            await server.generate_with_steering(message, sliders, ws, ws_id)
                        except Exception as e:
                            print(f"❌ Error generating: {e}")
                            await ws.send_json({
                                "type": "system",
                                "content": f"❌ Error: {str(e)}"
                            })
                            await ws.send_json({"type": "assistant_done"})
                    
                    elif msg_type == "clear":
                        # Clear conversation history
                        if ws_id in server.conversation_history:
                            history_len = len(server.conversation_history[ws_id])
                            server.conversation_history[ws_id] = []
                            print(f"🧹 Cleared {history_len} messages from conversation (ID: {ws_id[:8]}...)")
                            await ws.send_json({
                                "type": "system",
                                "content": "Conversation cleared. Starting fresh!"
                            })
                    
                except json.JSONDecodeError:
                    print("⚠️ Invalid JSON received")
                    
            elif msg.type == web.WSMsgType.ERROR:
                print(f"❌ WebSocket error: {ws.exception()}")
                
    except Exception as e:
        print(f"❌ Connection error: {e}")
    finally:
        # Clean up conversation history when client disconnects
        if ws_id in server.conversation_history:
            history_len = len(server.conversation_history[ws_id])
            del server.conversation_history[ws_id]
            print(f"🔌 Client disconnected (ID: {ws_id[:8]}..., cleared {history_len} messages)")
        else:
            print(f"🔌 Client disconnected (ID: {ws_id[:8]}...)")
        
    return ws


async def init_app():
    """Initialize the web application"""
    app = web.Application()
    app.router.add_get('/ws', websocket_handler)
    
    # Initialize model
    await server.initialize()
    
    return app


def main():
    """Run the server"""
    print("="*60)
    print("🚀 STEERING VECTOR WEBSOCKET SERVER")
    print("="*60)
    
    app = asyncio.run(init_app())
    
    print("\n✅ Server starting on http://localhost:8000")
    print("📡 WebSocket endpoint: ws://localhost:8000/ws")
    print("\nPress Ctrl+C to stop\n")
    
    web.run_app(app, host='0.0.0.0', port=8000)


if __name__ == "__main__":
    main()

