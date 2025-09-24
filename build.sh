#!/bin/bash

# Build the WASM package
echo "Building WASM package..."
wasm-pack build --target web

if [ $? -eq 0 ]; then
    echo "✅ WASM build successful!"
    echo "🚀 You can now serve the files with:"
    echo "   python3 -m http.server 8000"
    echo "   # or"
    echo "   npx serve ."
    echo ""
    echo "Then open http://localhost:8000 in your browser"
else
    echo "❌ WASM build failed!"
    exit 1
fi