#!/bin/bash
# Complete Atramentum workflow

echo "=== Atramentum Complete Workflow ==="
echo ""

# 1. Activate virtual environment
echo "Step 1: Activating virtual environment..."
source venv/bin/activate

# 2. Check for journal data
echo "Step 2: Checking for journal data..."
if [ ! -f "data/Journal.txt" ]; then
    echo "ERROR: No journal data found at data/Journal.txt"
    echo "Please add your journal export and try again"
    exit 1
fi

# 3. Process data
echo "Step 3: Processing journal data..."
python scripts/make_dataset.py --config configs/data_formatter.yaml

# 4. Create labels
echo "Step 4: Creating labels for RAG..."
python scripts/auto_label.py

# 5. Build index
echo "Step 5: Building search index..."
python scripts/build_index.py

# 6. Verify setup
echo "Step 6: Verifying setup..."
python test_complete.py

# 7. Optional: Start server
echo ""
echo "=== Setup Complete ==="
echo ""
echo "To start the API server:"
echo "  python scripts/serve_simple.py"
echo ""
echo "To train a model (requires GPU):"
echo "  python scripts/train_sft.py --config configs/sft_llama3.yaml"
echo ""
echo "To use Docker:"
echo "  docker compose -f docker/docker-compose.yml up"
