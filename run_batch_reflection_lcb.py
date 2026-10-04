"""
Batch reflection script for LiveCodeBench results.
Uses qwen_api_key_config.json to perform batch reflection on
samples.jsonl_results.jsonl (failed samples only).
"""

import sys
import time
from pathlib import Path

# Add src directory to Python path so we can import package modules
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from agents.batch_reflection_agent import BatchReflectionAgent


def main():
    config_path = SRC_DIR / "configs" / "qwen_api_key_config.json"
    results_file = (
        PROJECT_ROOT
        / "output"
        / "LiveCodeBench"
        / "qwen3-vl-8b-instruct-qwen3-vl-8b-instruct"
        / "samples.jsonl_results.jsonl"
    )
    data_dir = PROJECT_ROOT / "data" / "LiveCodeBench"

    print("=" * 60)
    print("Batch Reflection Configuration")
    print("=" * 60)
    print(f"config_path    : {config_path}")
    print(f"results_file   : {results_file}")
    print(f"data_dir       : {data_dir}")
    print(f"max_iterations : 3")
    print(f"reflection_mode: full")

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not results_file.exists():
        raise FileNotFoundError(f"Results file not found: {results_file}")
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    agent = BatchReflectionAgent(
        config_path=str(config_path),
        results_file=str(results_file),
        data_dir=str(data_dir),
        max_iterations=3,
        reflection_mode="full",
    )

    failed_samples = agent.get_failed_samples()
    print(f"\nFound {len(failed_samples)} failed samples to reflect.")

    if not failed_samples:
        print("No failed samples; nothing to do.")
        return

    overall_start = time.time()
    summary = []

    for idx, sample in enumerate(failed_samples, start=1):
        print(f"\n[{idx}/{len(failed_samples)}] Processing {sample.task_id}")
        try:
            iter_results = agent.process_sample(sample)
            for ir in iter_results:
                agent.save_single_iteration_result(ir)
                summary.append({
                    "task_id": ir.task_id,
                    "iteration": ir.iteration,
                    "issue_type": ir.issue_type,
                    "passed": ir.passed,
                    "duration": ir.duration,
                    "tokens": ir.token_usage,
                })
        except Exception as exc:  # noqa: BLE001
            print(f"!! Error processing {sample.task_id}: {exc}")
            continue

    elapsed = time.time() - overall_start
    passed_total = sum(1 for s in summary if s["passed"])
    print("\n" + "=" * 60)
    print("Batch Reflection Summary")
    print("=" * 60)
    print(f"Total iterations: {len(summary)}")
    print(f"Passed in reflection: {passed_total}")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Reflection output dir: {agent.output_dir}")


if __name__ == "__main__":
    main()
