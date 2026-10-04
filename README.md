# From Flowcharts to Code: Explicit Planning and Stage-Level Diagnosis for Multimodal Code Generation

Replication package for the paper. The framework recovers program logic from a
flowchart into an **executable Intermediate Logical Representation (ILR)**,
translates it into Python, and uses **ILR execution outcomes** to attribute a
failed test to either the planning stage or the implementation stage before
triggering targeted repair.

## Framework

Three agents cooperate in a revision loop (at most `r = 3` revision rounds):

- **Planner Agent** — extracts structured flowchart evidence (nodes, edges,
  connectivity) and constructs an ILR: a JSON graph of `start / process /
  decision / call / return / end` nodes whose decision nodes carry explicit
  `true_next` / `false_next` branch targets and process nodes carry executable
  state updates. Feedback-driven replanning reuses the cached structural
  evidence.
- **Coder Agent** — translates the ILR into Python that strictly realizes the
  ILR execution order while conforming to the task description and the starter
  code.
- **Critic Agent** — performs stage-level diagnosis on failures:
  - *Plan-level*: the ILR is executed in a sandboxed interpreter on the
    diagnostic tests. An ILR failure attributes the error to the planning
    stage; a label comparison against the OCR'd flowchart text (containment
    metric) and the flowchart image serve as localization references.
  - *Implementation-level*: if the ILR passes, the failing program is
    re-executed with instrumentation (executed statements, local variable
    states, function returns) and the bounded trace guides code repair.
  - *Safeguards*: the Critic may override the interpreter when it identifies
    no substantive plan issue; it may reattribute a failure to planning only
    by naming a specific ILR node together with a concrete counterexample
    (`ILR_LOGIC_ISSUE: node <id> | input: ... | ilr_output: ... | expected:
    ...`), accepted only after two deterministic checks — the claimed input
    must not coincide with a test case the interpreter already passed, and
    re-executing the ILR must not reproduce the claimed expected output.

## Benchmarks

| Benchmark | Subsets | Size | Notes |
|---|---|---|---|
| **Code-Vision** | HumanEval-V, Algorithm, MATH | 164 / 149 / 125 | Flowchart images + problem description + starter code; Easy/Medium/Hard splits on Algorithm and MATH |
| **LCB-Vision** | — | 148 | Image-augmented variant of LiveCodeBench: only problems annotated *hard* in all six release versions; flowcharts generated from official Python solutions via Gemini-3.1-Pro and verified |

Backbones: **Qwen3-VL-8B**, **Qwen3-VL-30B** (Qwen3-VL-30B-A3B-Instruct),
**Gemini-3.1-Flash-Lite**, **GPT-4o-mini**; **Gemini-3.1-Pro** is additionally
evaluated on LCB-Vision.

Compared methods: **Ours**, prompting baselines (**Direct**, **Zero-shot CoT**,
**Self-Planning**), and agent baselines (**LDB**, **MapCoder**).

## Project Structure

```
flowchart2code/
├── src/                              # Main framework + prompting baselines + evaluation
│   ├── main.py                       # Single-sample ILR pipeline entry
│   ├── run_all_datasets.py           # Batch generation + reflection with resume support (4 datasets)
│   ├── run_all_datasets.sh           # Shell launcher
│   ├── agents/
│   │   ├── langgraph_agent.py              # Base ILR→Code agent (LangGraph)
│   │   ├── agent1_ilr_generator_with_tools.py  # Planner: OCR tools + ILR construction
│   │   ├── agent2_code_generator.py            # Coder: ILR → Python
│   │   ├── batch_reflection_agent.py           # Critic + revision routing (full / code_only / ilr_only modes; also a CLI entry for Code-Vision reflection)
│   │   └── self_planning_prompt_utils.py       # Self-Planning prompts
│   ├── tools/
│   │   ├── ilr_interpreter.py            # Sandboxed ILR interpreter + tester (plan-level gate)
│   │   ├── code_tester.py                # unittest-style code execution + instrumentation
│   │   ├── lcb_tester.py                 # LiveCodeBench stdio/call-based tester
│   │   ├── ocr_extractor.py              # Flowchart OCR (cached in output/<dataset>/nodes.jsonl)
│   │   ├── flowchart_cache.py / problem_extractor.py
│   ├── evaluation/                   # pass@1 scoring (baseline / text / LCB)
│   ├── scripts/                      # Prompting-baseline batch runners (all modalities)
│   ├── human-eval/                   # Official HumanEval evaluator used for all pass/fail judgments
│   ├── utils/                        # API client, code utils, IR typing, merge & evaluate
│   ├── session/                      # Resume/session management
│   └── configs/                      # Model API configurations (see below)
├── data/
│   ├── HumanEval-V/  Algorithm/  MATH/      # Code-Vision (images + problems + tests + *_with_sample_io.jsonl)
│   └── LiveCodeBench/                        # LCB-Vision (images + problems; LiveCodeBench_merged.jsonl
│                                             #   is the evaluation source, *_with_sample_io.jsonl the public tests)
├── output/
│   ├── Algorithm/  HumanEval-V/  MATH/      # Code-Vision runs, one folder per <model[-variant]>
│   ├── LiveCodeBench/                        # LCB-Vision runs
│   ├── failed/                       # RQ2 stage-attribution controlled set (100 samples)
│   ├── stage_analysis/  stage_analysis_gpt/  # Per-problem stage-diagnosis artifacts (RQ2)
│   ├── MapCoder/                     # MapCoder agent-baseline results
│   ├── LDB/                          # LDB agent-baseline results
│   ├── progress/                     # Resume checkpoints for batch runs
│   └── *_codevision_results.txt      # Combined pass@1 summary per experiment (all datasets)
├── lcb_batch_reflection.py           # LCB-Vision reflection runner (LiveCodeBench-specific)
├── run_batch_reflection_lcb.py       # LCB-Vision reflection launcher
├── run_all_evaluations.sh            # Unified runner: baselines + framework + ablations
└── requirements.txt
```

## Installation

```bash
conda create -n flowchart2code python==3.10
conda activate flowchart2code
pip install -r requirements.txt
```

## Configuration

Create the model configuration files in `src/configs/` from the provided
template `api_key_config.example.json` (config files with credentials are not
committed). The framework expects the following files:

| File | Model | Usage |
|---|---|---|
| `qwen_api_key_config_deploy.json` | Qwen3-VL-8B (local vLLM, OpenAI-compatible) | Code-Vision + LCB-Vision, all stages |
| `qwen_api_key_config.json` | Qwen3-VL-8B (alternate endpoint) | — |
| `qwen3_vl_30_config.json` | Qwen3-VL-30B | Code-Vision + LCB-Vision |
| `gemini_api_key_config.json` | Gemini-3.1-Flash-Lite | Code-Vision + LCB-Vision |
| `gpt_api_key_config.json` | GPT-4o-mini | Code-Vision + LCB-Vision |
| `gemini3.1p_api_key_config.json` | Gemini-3.1-Pro | LCB-Vision |

Common format:

```json
{
    "api_key": "YOUR_API_KEY",
    "base_url": "https://.../v1",
    "model": "model-name",
    "max_tokens": 20000,
    "temperature": 0.2,
    "top_p": 0.95,
    "n": 1,
    "stop": []
}
```

The Qwen3-VL-8B backbone is served locally with vLLM (OpenAI-compatible
interface, tensor-parallel size 2, max context 128K).

## Running the Framework (Ours)

Initial generation (Planner → Coder) and reflection (Critic + targeted repair)
are decoupled for engineering efficiency: reflection processes only the failed
samples of a run.

```bash
cd src
# 1) Initial generation (ILR pipeline), Code-Vision and LCB-Vision alike
python3 run_all_datasets.py \
    --vision-model-2 qwen_api_key_config_deploy.json \
    --language-model-1 qwen_api_key_config_deploy.json \
    --intermediate-representation ilr \
    --include-problem-text-description true \
    --data-root ../data --output-dir ../output

# 2) Reflection (Critic + stage-level repair), Code-Vision:
#    batch_reflection_agent.py reads the results file of step 1 and repairs
#    the failed samples for up to 3 revision rounds.
python3 agents/batch_reflection_agent.py \
    --config configs/qwen_api_key_config_deploy.json \
    --results ../output/<dataset>/<model>-<model>/samples.jsonl_results.jsonl \
    --data-dir ../data/<dataset> \
    --output-dir ../output/<dataset>/<model>-<model>/reflection \
    --max-iterations 3 \
    --reflection-mode full \
    --intermediate-representation ilr
```

The unified script runs initial generation and the prompting baselines:

```bash
bash run_all_evaluations.sh --datasets HumanEval-V,Algorithm,MATH,LiveCodeBench
```

### Ablations

| Variant | How to run |
|---|---|
| w/o ILR (free-form textual plan) | `run_all_datasets.py --intermediate-representation text` (initial generation), then reflection with `--intermediate-representation text` |
| w/o iteration | reflection with `--max-iterations 1` |
| w/o stage isolation (joint diagnosis) | joint ILR+code analysis variant used for the paper's Table 5; not included in this package (the shipped Critic is the staged version) |
| w/o implementation-level diagnosis | reflection with `--reflection-mode ilr_only` |
| Restricted feedback (public tests only) | reflection error context is drawn from `<dataset>/*_with_sample_io.jsonl` public cases; hidden-test details are withheld automatically when all public cases pass |

### Output layout

```
output/<dataset>/<model>/                     # Direct baseline
output/<dataset>/<model>_zero_shot_cot/       # Zero-shot CoT
output/<dataset>/<model>_self_planning/       # Self-Planning
output/<dataset>/<model>_text_image/          # Image + text direct (modality study)
output/<dataset>/<model>-text/                # Text-only direct (modality study)
output/<dataset>/<model>-<model>/             # Ours: initial ILR pipeline
output/<dataset>/<model>-<model>/reflection/  # Ours: revision iterations (ilr1.jsonl, results_iteration.txt, ...)
output/<dataset>/<model>-<model>-text-ir/     # w/o ILR ablation
```

Each run folder contains `samples.jsonl` (generations), `ilr.jsonl` (ILRs),
`samples.jsonl_results.jsonl` (test results), `results.txt` (pass@1 summary)
and `token_stats.json`. `1`/`2`-prefixed folders are the repeated full-pipeline
runs used for the stability analysis.

## Baselines

**Prompting baselines** (Direct, Zero-shot CoT, Self-Planning, and the
modality study) share the framework's input information and are run via:

```bash
cd src
python3 scripts/evaluate_all_datasets.py \
    --api-config gpt_api_key_config.json \
    --prompt-variant default|zero_shot_cot|self_planning \
    [--text-image] \
    --data-root ../data --output-dir ../output

# Text-only (no flowchart image)
python3 scripts/evaluate_all_datasets_text.py \
    --api-config gpt_api_key_config.json \
    --data-root ../data --output-dir ../output
```

**Agent baselines** (LDB, MapCoder) were run with their original codebases;
their final per-model results are kept under `output/LDB/` and
`output/MapCoder/`.

## LCB-Vision

```bash
# 1) Initial generation: run_all_datasets.py registers LiveCodeBench as a dataset
cd src && python3 run_all_datasets.py --data-root ../data --output-dir ../output ...

# 2) Reflection over the LCB-Vision results (stdio/call-based testing via tools/lcb_tester.py)
python3 run_batch_reflection_lcb.py        # launcher (edit model config / paths inside)
# or directly: python3 lcb_batch_reflection.py --help
```

## Stage-Attribution Controlled Set (RQ2)

`output/failed/` contains the balanced controlled set used for the stage
attribution study (see `output/failed/README.md` for details): 100 samples
from the Qwen3-VL-8B failure pool, where each of the 50 plan-error samples
additionally has a *Code-error* variant (ILR repaired to pass all sandboxed
interpreter tests, code translated from the repaired plan with exactly one
injected implementation bug), giving unambiguous ground truth for both error
types. The per-problem stage-diagnosis artifacts are kept in
`output/stage_analysis*/`, and per-experiment combined pass@1 summaries in
`output/*_codevision_results.txt`.

## Evaluation

All generated programs are executed with the official HumanEval evaluator
(`src/human-eval/`) against benchmark-provided test cases; **pass@1** with one
final program per problem is the primary metric. LCB-Vision programs are
judged by the LiveCodeBench-style tester (`src/tools/lcb_tester.py`).
Token consumption over the entire workflow (generation + all revision
interactions) is recorded per run in `token_stats.json` and `results.txt`.
