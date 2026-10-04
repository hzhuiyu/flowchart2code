#!/bin/bash

# Batch evaluation script (LangGraph multi-agent workflow)
# Supports running three datasets based on the provided config, evaluating and saving results after each dataset completes
# Supports checkpoint resume functionality

set -e  # Exit on error

# Get the absolute path of the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Default configuration (can be modified to your configuration)
VISION_MODEL_2_CONFIG="qwen2_5_vl_7b_local_config.json"  # For pseudocode generation (locally deployed Qwen2.5-VL-7B)
LANGUAGE_MODEL_1_CONFIG="qwen_code_config.json"   # For code generation
INTERMEDIATE_REPRESENTATION="ilr"
INCLUDE_PROBLEM_TEXT_DESCRIPTION="true"

# Data root directory
DATA_ROOT="$PROJECT_ROOT/data"

# Output directory
OUTPUT_DIR="$PROJECT_ROOT/output"

# Help information
function show_help() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --vision-model-2 <config>  Vision model 2 config file (for pseudocode generation)"
    echo "  --language-model-1 <config> Language model 1 config file (for code generation)"
    echo "  --data-root <dir>         Data root directory (default: $DATA_ROOT)"
    echo "  --output-dir <dir>        Output directory (default: $OUTPUT_DIR)"
    echo "  --intermediate-representation <type> Agent1 intermediate representation type (ilr/text)"
    echo "  --include-problem-text-description <true|false> Whether to include problem text description (default: true)"
    echo "  --reset                   Reset progress and restart evaluation"
    echo "  -h, --help                Show help information"
    echo ""
    echo "Examples:"
    echo "  # Run with default configuration"
    echo "  $0"
    echo ""
    echo "  # Run with specified configuration"
    echo "  $0 --vision-model-2 glm_api_key_config.json --language-model-1 glm_api_key_config.json"
    echo ""
    echo "  # Reset progress and rerun"
    echo "  $0 --reset"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vision-model-2)
            VISION_MODEL_2_CONFIG="$2"
            shift 2
            ;;
        --language-model-1)
            LANGUAGE_MODEL_1_CONFIG="$2"
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
        --intermediate-representation)
            INTERMEDIATE_REPRESENTATION="$2"
            shift 2
            ;;
        --include-problem-text-description)
            INCLUDE_PROBLEM_TEXT_DESCRIPTION="$2"
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

# Check if configuration files exist
VISION_CONFIG_PATH="$SCRIPT_DIR/configs/$VISION_MODEL_2_CONFIG"
LANGUAGE_CONFIG_PATH="$SCRIPT_DIR/configs/$LANGUAGE_MODEL_1_CONFIG"

if [ ! -f "$VISION_CONFIG_PATH" ]; then
    echo "Error: Vision model config file does not exist: $VISION_CONFIG_PATH"
    exit 1
fi

if [ ! -f "$LANGUAGE_CONFIG_PATH" ]; then
    echo "Error: Language model config file does not exist: $LANGUAGE_CONFIG_PATH"
    exit 1
fi

# Display configuration information
echo "=========================================="
echo "Batch Evaluation (LangGraph Multi-Agent Workflow)"
echo "=========================================="
echo "Vision model 2 config: $VISION_MODEL_2_CONFIG"
echo "Language model 1 config: $LANGUAGE_MODEL_1_CONFIG"
echo "Intermediate representation type: $INTERMEDIATE_REPRESENTATION"
echo "Problem text description: $INCLUDE_PROBLEM_TEXT_DESCRIPTION"
echo "Data root directory: $DATA_ROOT"
echo "Output directory: $OUTPUT_DIR"
echo "=========================================="
echo ""

# Run Python script
cd "$SCRIPT_DIR"

python3 run_all_datasets.py \
    --vision-model-2 "$VISION_MODEL_2_CONFIG" \
    --language-model-1 "$LANGUAGE_MODEL_1_CONFIG" \
    --intermediate-representation "$INTERMEDIATE_REPRESENTATION" \
    --include-problem-text-description "$INCLUDE_PROBLEM_TEXT_DESCRIPTION" \
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
