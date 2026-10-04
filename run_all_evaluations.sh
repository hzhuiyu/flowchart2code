#!/bin/bash
# =============================================================================
# run_all_evaluations.sh
# Unified evaluation script: automatically runs baseline, ILR pipeline, reflection agent, etc.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

# ========================= Configuration =========================
# Set the model config files to test here (placed under src/configs/)

# Baseline (flowchart → code direct generation) uses a single config
BASELINE_CONFIGS=(
    "gpt_api_key_config.json"
    # "openai_api_key_config.json"
    # "gemini_api_key_config.json"
    # "claude_api_key_config.json"
)

# Configs used by ILR pipeline
# vision_model_2 for ILR generation (vision model)
# language_model_1 for code generation (language model)
ILR_VISION_MODEL_2="openai_api_key_config.json"
ILR_LANGUAGE_MODEL_1="qwen_api_key_config.json"

# Config used by reflection agent (same as ILR pipeline vision/language)
REFLECTION_MODEL="qwen_api_key_config.json"

# Datasets (leave empty to run all)
DATASETS=()
# To run specific datasets only, uncomment:
# DATASETS=("HumanEval-V")
# DATASETS=("Algorithm" "MATH")

# Intermediate representation type: ilr or text
INTERMEDIATE_REPRESENTATION="ilr"

# Whether to include problem text description
INCLUDE_PROBLEM_TEXT_DESCRIPTION="true"

# Data/output paths
DATA_ROOT="$PROJECT_ROOT/data"
OUTPUT_DIR="$PROJECT_ROOT/output"

# Which experiments to run (set to true to enable)
RUN_BASELINE_DEFAULT=true
RUN_BASELINE_ZERO_SHOT_COT=true
RUN_BASELINE_SELF_PLANNING=true
RUN_BASELINE_TEXT_IMAGE=true
RUN_ILR_PIPELINE=true
RUN_ILR_REFLECTION=true
RUN_CODE_ONLY=true

# ========================= Functions =========================

usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  --baseline-config <config>   Append baseline config file"
    echo "  --ilr-vision <config>        ILR vision model config"
    echo "  --ilr-language <config>      ILR language model config"
    echo "  --reflection-model <config> Reflection model config"
    echo "  --datasets <name1,name2>     Specify datasets (comma-separated)"
    echo "  --output-dir <dir>           Output directory"
    echo "  --skip-baseline              Skip baseline experiments"
    echo "  --skip-ilr                   Skip ILR pipeline experiments"
    echo "  --skip-reflection            Skip reflection experiments"
    echo "  --only-baseline              Run only baseline"
    echo "  --only-ilr                   Run only ILR pipeline + reflection"
    echo "  -h, --help                   Show help"
    echo ""
    echo "Examples:"
    echo "  $0                                    # Run all experiments"
    echo "  $0 --only-baseline                    # Run baseline only"
    echo "  $0 --only-ilr                         # Run ILR + reflection only"
    echo "  $0 --datasets HumanEval-V,Algorithm   # Specify datasets"
}

build_dataset_args() {
    local args=""
    for d in "${DATASETS[@]}"; do
        args="$args --dataset $d"
    done
    echo "$args"
}

# ========================= Parse Arguments =========================
while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline-config)
            BASELINE_CONFIGS+=("$2"); shift 2 ;;
        --ilr-vision)
            ILR_VISION_MODEL_2="$2"; shift 2 ;;
        --ilr-language)
            ILR_LANGUAGE_MODEL_1="$2"; shift 2 ;;
        --reflection-model)
            REFLECTION_MODEL="$2"; shift 2 ;;
        --datasets)
            IFS=',' read -ra DATASETS <<< "$2"; shift 2 ;;
        --output-dir)
            OUTPUT_DIR="$2"; shift 2 ;;
        --skip-baseline)
            RUN_BASELINE_DEFAULT=false
            RUN_BASELINE_ZERO_SHOT_COT=false
            RUN_BASELINE_SELF_PLANNING=false
            RUN_BASELINE_TEXT_IMAGE=false
            RUN_CODE_ONLY=false
            shift ;;
        --skip-ilr)
            RUN_ILR_PIPELINE=false
            shift ;;
        --skip-reflection)
            RUN_ILR_REFLECTION=false
            shift ;;
        --only-baseline)
            RUN_ILR_PIPELINE=false
            RUN_ILR_REFLECTION=false
            shift ;;
        --only-ilr)
            RUN_BASELINE_DEFAULT=false
            RUN_BASELINE_ZERO_SHOT_COT=false
            RUN_BASELINE_SELF_PLANNING=false
            RUN_BASELINE_TEXT_IMAGE=false
            RUN_CODE_ONLY=false
            shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            echo "Unknown option: $1"; usage; exit 1 ;;
    esac
done

DATASET_ARGS=$(build_dataset_args)

echo "=========================================="
echo "  Flowchart2Code Unified Evaluation"
echo "=========================================="
echo "Baseline configs:  ${BASELINE_CONFIGS[*]}"
echo "ILR vision model:  $ILR_VISION_MODEL_2"
echo "ILR language model: $ILR_LANGUAGE_MODEL_1"
echo "Reflection model:   $REFLECTION_MODEL"
echo "Datasets:           ${DATASETS[*]:-all}"
echo "Output dir:         $OUTPUT_DIR"
echo "=========================================="

# ========================= Baseline Experiments =========================

for CONFIG in "${BASELINE_CONFIGS[@]}"; do
    CONFIG_PATH="$SCRIPT_DIR/src/configs/$CONFIG"
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "⚠ Config file not found: $CONFIG_PATH, skipping"
        continue
    fi

    MODEL_NAME=$(python3 -c "import json; print(json.load(open('$CONFIG_PATH'))['model'])" 2>/dev/null || echo "unknown")
    echo ""
    echo "###### Baseline: $CONFIG ($MODEL_NAME) ######"

    # 1. Baseline default (flowchart -> code)
    if [ "$RUN_BASELINE_DEFAULT" = true ]; then
        echo ""
        echo "--- [1/5] Baseline: default (flowchart -> code) ---"
        cd "$SCRIPT_DIR"
        python3 src/scripts/evaluate_all_datasets.py \
            --api-config "$CONFIG" \
            --prompt-variant default \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR" \
            $DATASET_ARGS || echo "⚠ Baseline default failed"
    fi

    # 2. Baseline Zero-Shot-CoT
    if [ "$RUN_BASELINE_ZERO_SHOT_COT" = true ]; then
        echo ""
        echo "--- [2/5] Baseline: Zero-Shot-CoT ---"
        cd "$SCRIPT_DIR"
        python3 src/scripts/evaluate_all_datasets.py \
            --api-config "$CONFIG" \
            --prompt-variant zero_shot_cot \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR" \
            $DATASET_ARGS || echo "⚠ Baseline Zero-Shot-CoT failed"
    fi

    # 3. Baseline Self-Planning (two-stage: plan then generate)
    if [ "$RUN_BASELINE_SELF_PLANNING" = true ]; then
        echo ""
        echo "--- [3/5] Baseline: Self-Planning ---"
        cd "$SCRIPT_DIR"
        python3 src/scripts/evaluate_all_datasets.py \
            --api-config "$CONFIG" \
            --prompt-variant self_planning \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR" \
            $DATASET_ARGS || echo "⚠ Baseline Self-Planning failed"
    fi

    # 4. Baseline text+image (text description + image)
    if [ "$RUN_BASELINE_TEXT_IMAGE" = true ]; then
        echo ""
        echo "--- [4/5] Baseline: Text+Image ---"
        cd "$SCRIPT_DIR"
        python3 src/scripts/evaluate_all_datasets.py \
            --api-config "$CONFIG" \
            --prompt-variant default \
            --text-image \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR" \
            $DATASET_ARGS || echo "⚠ Baseline Text+Image failed"
    fi

    # 5. Code-only (text only, no image)
    if [ "$RUN_CODE_ONLY" = true ]; then
        echo ""
        echo "--- [5/5] Code-Only (text-only, no image) ---"
        cd "$SCRIPT_DIR"
        python3 src/scripts/evaluate_all_datasets_text.py \
            --api-config "$CONFIG" \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR" \
            $DATASET_ARGS || echo "⚠ Code-Only failed"
    fi
done

# ========================= ILR Pipeline Experiments =========================

ILR_VISION_PATH="$SCRIPT_DIR/src/configs/$ILR_VISION_MODEL_2"
ILR_LANGUAGE_PATH="$SCRIPT_DIR/src/configs/$ILR_LANGUAGE_MODEL_1"

if [ "$RUN_ILR_PIPELINE" = true ]; then
    if [ ! -f "$ILR_VISION_PATH" ] || [ ! -f "$ILR_LANGUAGE_PATH" ]; then
        echo "⚠ ILR pipeline config files missing, skipping"
    else
        echo ""
        echo "###### ILR Pipeline: vision=$ILR_VISION_MODEL_2, language=$ILR_LANGUAGE_MODEL_1 ######"

        echo ""
        echo "--- ILR Pipeline: generation + evaluation (LangGraph multi-agent) ---"
        cd "$SCRIPT_DIR/src"
        python3 run_all_datasets.py \
            --vision-model-2 "$ILR_VISION_MODEL_2" \
            --language-model-1 "$ILR_LANGUAGE_MODEL_1" \
            --intermediate-representation "$INTERMEDIATE_REPRESENTATION" \
            --include-problem-text-description "$INCLUDE_PROBLEM_TEXT_DESCRIPTION" \
            --data-root "$DATA_ROOT" \
            --output-dir "$OUTPUT_DIR" \
            $DATASET_ARGS || echo "⚠ ILR Pipeline failed"
    fi
fi

# ========================= Reflection Agent Experiments =========================

REFLECTION_PATH="$SCRIPT_DIR/src/configs/$REFLECTION_MODEL"

if [ "$RUN_ILR_REFLECTION" = true ]; then
    if [ ! -f "$ILR_VISION_PATH" ] || [ ! -f "$ILR_LANGUAGE_PATH" ] || [ ! -f "$REFLECTION_PATH" ]; then
        echo "⚠ Reflection agent config files missing, skipping"
    else
        echo ""
        echo "###### Reflection Agent: config=$REFLECTION_MODEL ######"

        # Model run folder name, matching run_all_datasets._get_model_name:
        # "<clean(vision_model)>-<clean(language_model)>" + IR suffix
        MODEL_RUN_NAME=$(cd "$SCRIPT_DIR" && python3 -c "
import json, re, sys
sys.path.insert(0, 'src')
from utils.intermediate_representation import get_intermediate_representation_suffix
def clean(n):
    return re.sub(r'-+', '-', re.sub(r'[^a-zA-Z0-9-]', '-', n)).strip('-')
v = json.load(open(sys.argv[1], encoding='utf-8'))['model']
l = json.load(open(sys.argv[2], encoding='utf-8'))['model']
print(f'{clean(v)}-{clean(l)}' + get_intermediate_representation_suffix(sys.argv[3]))
" "$ILR_VISION_PATH" "$ILR_LANGUAGE_PATH" "$INTERMEDIATE_REPRESENTATION")

        # Datasets to reflect over (default: all four)
        if [ ${#DATASETS[@]} -eq 0 ]; then
            REFLECT_DATASETS=("HumanEval-V" "Algorithm" "MATH" "LiveCodeBench")
        else
            REFLECT_DATASETS=("${DATASETS[@]}")
        fi

        for ds in "${REFLECT_DATASETS[@]}"; do
            RESULTS_FILE="$OUTPUT_DIR/$ds/$MODEL_RUN_NAME/samples.jsonl_results.jsonl"
            if [ ! -f "$RESULTS_FILE" ]; then
                echo "⚠ No results file for $ds ($RESULTS_FILE); run the ILR pipeline first, skipping"
                continue
            fi
            echo ""
            echo "--- Reflection Agent on $ds (model run: $MODEL_RUN_NAME) ---"
            cd "$SCRIPT_DIR/src"
            python3 agents/batch_reflection_agent.py \
                --config "configs/$REFLECTION_MODEL" \
                --results "$RESULTS_FILE" \
                --data-dir "$DATA_ROOT/$ds" \
                --output-dir "$OUTPUT_DIR/$ds/$MODEL_RUN_NAME/reflection" \
                --max-iterations 3 \
                --reflection-mode full \
                --intermediate-representation "$INTERMEDIATE_REPRESENTATION" \
                || echo "⚠ Reflection on $ds failed"
        done
    fi
fi

echo ""
echo "=========================================="
echo "  ✓ All evaluations completed"
echo "=========================================="
