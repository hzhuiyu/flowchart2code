#!/bin/bash

# Universal batch evaluation script for src
# Supports automatic evaluation of three datasets (HumanEval-V, Algorithm, MATH) based on the provided config
# Each dataset is evaluated and results are saved immediately after completion

set -e  # Exit on error

# Get the absolute path of the script's directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Default config (can be modified to your config)
API_CONFIG="qwen_vl_api_key_config.json"

# Data root directory
DATA_ROOT="$PROJECT_ROOT/data"

# Output directory
OUTPUT_DIR="$PROJECT_ROOT/output"

# Help message
function show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --api-config <config>     API config file"
    echo "  --data-root <dir>         Data root directory (default: $DATA_ROOT)"
    echo "  --output-dir <dir>        Output directory (default: $OUTPUT_DIR)"
    echo "  --reset                   Reset progress and restart evaluation"
    echo "  -h, --help                Show help message"
    echo ""
    echo "Examples:"
    echo "  # Run with default config"
    echo "  $0"
    echo ""
    echo "  # Run with GLM config"
    echo "  $0 --api-config glm_api_key_config.json"
    echo ""
    echo "  # Reset progress and rerun"
    echo "  $0 --reset"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-config)
            API_CONFIG="$2"
            shift 2
            ;;
        --data-root)
            DATA_ROOT="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --reset)
            RESET="--reset"
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if config file exists
CONFIG_PATH="$SCRIPT_DIR/../configs/$API_CONFIG"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: API config file not found: $CONFIG_PATH"
    exit 1
fi

# Display configuration info
echo "=========================================="
echo "src Universal Batch Evaluation"
echo "=========================================="
echo "API Config: $API_CONFIG"
echo "Data Root: $DATA_ROOT"
echo "Output Dir: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run Python script
cd "$SCRIPT_DIR"

python3 evaluate_all_datasets.py \
    --api-config "$API_CONFIG" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_DIR" \
    $RESET

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✓ Batch evaluation completed successfully"
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "✗ Batch evaluation failed"
    echo "=========================================="
    exit 1
fi
