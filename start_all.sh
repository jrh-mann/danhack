#!/bin/bash
# Start both backend and frontend servers

echo "🚀 Starting Steering Vector Chatbot..."
echo ""

# Check for API key
if [ -z "$OPENROUTER_API_KEY" ]; then
    echo "❌ Error: OPENROUTER_API_KEY not set"
    echo "   Run: export OPENROUTER_API_KEY='your-key-here'"
    exit 1
fi

# Activate venv and start backend in background
echo "📡 Starting backend server..."
source .venv/bin/activate
python server.py &
BACKEND_PID=$!

# Wait for backend to initialize
sleep 3

# Check if npm is available
if command -v npm &> /dev/null; then
    # Start frontend
    echo "🎨 Starting frontend..."
    cd frontend
    npm run dev &
    FRONTEND_PID=$!
    
    echo ""
    echo "✅ Servers running:"
    echo "   Backend:  http://localhost:8000"
    echo "   Frontend: http://localhost:5173"
    echo ""
    echo "Press Ctrl+C to stop both servers"
    
    # Trap Ctrl+C and kill both processes
    trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
    
    # Wait for both processes
    wait
else
    echo ""
    echo "⚠️  npm not found - starting backend only"
    echo ""
    echo "✅ Backend running: http://localhost:8000"
    echo ""
    echo "To start frontend separately:"
    echo "   cd frontend && npm run dev"
    echo ""
    echo "To install Node.js/npm:"
    echo "   Ubuntu/Debian: sudo apt install nodejs npm"
    echo "   Or use nvm: https://github.com/nvm-sh/nvm"
    echo ""
    echo "Press Ctrl+C to stop backend server"
    
    # Trap Ctrl+C and kill backend process
    trap "kill $BACKEND_PID 2>/dev/null; exit" INT
    
    # Wait for backend process
    wait $BACKEND_PID
fi

