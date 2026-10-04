# LDB on LiveCodeBench (via flowchart2code's converted data + mixed-I/O evaluator).
#
# 1. Convert the dataset (only needed once):
#      cd programming && python lcb_convert.py
#
# 2. (Optional) Generate seeds with LDB's simple strategy:
#      ./run_ldb_lcb.sh --seed [model] [output_dir]
#
# 3. Debug with LDB, using an existing seed file:
#      ./run_ldb_lcb.sh [model] [output_dir] [seedfile]
#
# Model backends:
#   gpt-4*/gpt-5*/o3*/o4*  -> OpenAI API (OPENAI_API_KEY env)
#   anything else          -> OpenAI-compatible endpoint via
#                             LDB_OPENAI_BASE_URL / LDB_OPENAI_API_KEY
#   LDB_MAX_CONTEXT_TOKENS raises the chat truncation budget (default 3097).
dataset=livecodebench

# Default LLM backend: flowchart2code's OpenAI-compatible config
# (api_key/base_url/model/max_tokens).  Pass --model config to use the
# config's model name, or unset LDB_CONFIG_PATH to fall back to plain OpenAI.
export LDB_CONFIG_PATH="${LDB_CONFIG_PATH:-../../src/configs/gpt_api_key_config.json}"

if [ "$1" = "--seed" ]; then
  model=$2
  output_dir=$3
  python main.py \
    --run_name "$output_dir" \
    --root_dir "../output_data/simple/$dataset/$model/" \
    --dataset_path "../input_data/$dataset/dataset/probs.jsonl" \
    --strategy simple \
    --model "$model" \
    --n_proc "1" \
    --testfile "../input_data/$dataset/test/tests.jsonl" \
    --verbose \
    --port "8000"
  exit 0
fi

model=$1
output_dir=$2
seedfile=$3
strategy="ldb"
python main.py \
  --run_name "$output_dir" \
  --root_dir "../output_data/$strategy/$dataset/$model/" \
  --dataset_path "../input_data/$dataset/dataset/probs.jsonl" \
  --strategy "$strategy" \
  --model "$model" \
  --seedfile "$seedfile" \
  --pass_at_k "1" \
  --max_iters "10" \
  --n_proc "1" \
  --port "8000" \
  --testfile "../input_data/$dataset/test/tests.jsonl" \
  --verbose
