#!/bin/bash

echo "🚀 Testing UI Improvements..."
echo ""

# Check if server is running
echo "1️⃣ Checking API server..."
if curl -s http://localhost:8000/health > /dev/null 2>&1; then
    echo "   ✅ API server is running"
else
    echo "   ❌ API server not running. Start with:"
    echo "      uvicorn app.api:app --reload --host 0.0.0.0 --port 8000"
    exit 1
fi

echo ""
echo "2️⃣ Checking frontend files..."
if [ -f "frontend/index.html" ]; then
    echo "   ✅ index.html exists"
    
    # Check if textarea was removed
    if grep -q "feedbackText" frontend/index.html; then
        echo "   ⚠️  Warning: feedbackText still in HTML (should be removed)"
    else
        echo "   ✅ Text feedback removed"
    fi
fi

if [ -f "frontend/script.js" ]; then
    echo "   ✅ script.js exists"
    
    # Check if autoRefineSearch exists
    if grep -q "autoRefineSearch" frontend/script.js; then
        echo "   ✅ Auto-refine function added"
    else
        echo "   ⚠️  Warning: autoRefineSearch not found"
    fi
fi

if [ -f "frontend/styles_improved.css" ]; then
    echo "   ✅ styles_improved.css exists"
    
    # Check if tag styles exist
    if grep -q "tag-color" frontend/styles_improved.css; then
        echo "   ✅ Metadata tag styles added"
    else
        echo "   ⚠️  Warning: tag styles not found"
    fi
fi

echo ""
echo "3️⃣ Opening frontend in browser..."
if command -v xdg-open > /dev/null 2>&1; then
    xdg-open http://localhost:8000/
elif command -v open > /dev/null 2>&1; then
    open http://localhost:8000/
else
    echo "   ℹ️  Please manually open: http://localhost:8000/"
fi

echo ""
echo "✨ Testing Checklist:"
echo "   [ ] Search for 'cat' or 'dog'"
echo "   [ ] Click 👍 on 2 images"
echo "   [ ] See counter: '(1 more to auto-refine)'"
echo "   [ ] Click 👍 on 3rd image"
echo "   [ ] See '⏳ Refining results...'"
echo "   [ ] Results should auto-update!"
echo "   [ ] Check if caption and tags show"
echo ""
echo "🎉 Done! Enjoy the improved UX!"
