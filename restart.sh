#!/bin/bash
# Restart XLA Image Retrieval API

echo "🔄 Restarting XLA API..."

# Stop existing process
pkill -f "uvicorn app.api:app"
sleep 2

# Start new process
cd /root/XLA/Image-Retrieval-XlaProject
source .venv39/bin/activate
nohup uvicorn app.api:app --host 0.0.0.0 --port 8000 > /var/log/xla-api.log 2>&1 &

sleep 3

# Check if running
if ps aux | grep -v grep | grep "uvicorn app.api:app" > /dev/null; then
    echo "✅ API restarted successfully"
    echo "📋 Log: tail -f /var/log/xla-api.log"
else
    echo "❌ Failed to start API"
    echo "Check log: tail -50 /var/log/xla-api.log"
    exit 1
fi
