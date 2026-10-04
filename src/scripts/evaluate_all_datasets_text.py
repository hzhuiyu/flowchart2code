#!/usr/bin/env python3
"""
Text-to-Code Batch Evaluation Script
=====================================
Input: Natural language problem description + starter_code (no image)
Supports three datasets: HumanEval-V, Algorithm, MATH
Pass a single api_config to run all datasets
"""

import sys
import os
import json
import argparse
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Any
from tqdm import tqdm

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluate_text import (
    read_jsonl_file,
    write_jsonl_file,
    extract_code,
    construct_prompt_text,
    construct_prompt_cot_text,
    get_response_text,
    calculate_token_statistics,
)
from evaluation.evaluate_lcb import evaluate_lcb_samples, save_lcb_results


class UniversalBatchTextEvaluator:
    """Pure text batch evaluator (no image)"""

    DATASETS = [
        {
            "name": "HumanEval-V",
            "data_file": "HumanEval.jsonl",
        },
        {
            "name": "Algorithm",
            "data_file": "Algorithm.jsonl",
        },
        {
            "name": "MATH",
            "data_file": "MATH.jsonl",
        },
        {
            "name": "LiveCodeBench",
            "data_file": "LiveCodeBench.jsonl",
            "original_data_file": "LiveCodeBench_original.jsonl",  # Original LCB data with test cases
        },
    ]

    def __init__(self, api_config: str, data_root: str = None, output_dir: str = None,
                 prompt_variant: str = "default", enable_stream_output: bool = False):
        self.api_config = self._load_config(api_config)

        if data_root is None:
            data_root = str(Path(__file__).parent.parent.parent / "data")
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent.parent / "output")

        self.data_root = data_root
        self.output_dir = output_dir
        self.prompt_variant = (prompt_variant or "default").strip().lower()

        self.model_name = self.api_config.get("model", "model").replace("/", "-")
        if self.prompt_variant == "cot":
            self.model_name += "_cot"

        self.progress_file = Path(self.output_dir) / "progress" / f"{self.model_name}_text_progress.json"
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)

        self.enable_stream_output = enable_stream_output
        if enable_stream_output:
            self.api_config["stream"] = True

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        if not Path(config_path).exists():
            config_path = Path(__file__).parent.parent / "configs" / config_path
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_progress(self) -> Dict[str, Any]:
        if not self.progress_file.exists():
            return {"model_name": self.model_name, "datasets": {}, "last_update": None}
        with open(self.progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_progress(self, progress: Dict[str, Any]) -> None:
        progress["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)

    def _get_data_hash(self, dataset: Dict[str, Any]) -> str:
        data_file = Path(self.data_root) / dataset["name"] / dataset["data_file"]
        if not data_file.exists():
            return None
        with open(data_file, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()

    def _check_completed(self, dataset_name: str, data_hash: str) -> bool:
        progress = self._load_progress()
        if dataset_name not in progress["datasets"]:
            return False
        if progress["datasets"][dataset_name].get("hash") != data_hash:
            print(f"  Dataset {dataset_name} has changed, needs re-evaluation")
            return False
        return progress["datasets"][dataset_name].get("completed", False)

    def _update_progress(self, dataset_name: str, data_hash: str,
                          status: str, message: str = None) -> None:
        progress = self._load_progress()
        if dataset_name not in progress["datasets"]:
            progress["datasets"][dataset_name] = {"hash": data_hash, "status": status,
                                                   "completed": False, "messages": []}
        dp = progress["datasets"][dataset_name]
        dp["hash"] = data_hash
        dp["status"] = status
        dp["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        if status == "completed":
            dp["completed"] = True
        if message:
            dp["messages"].append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "message": message})
        self._save_progress(progress)

    def _output_dir_for(self, dataset_name: str) -> Path:
        return Path(self.output_dir) / dataset_name / f"{self.model_name}-text"

    def _samples_path(self, dataset_name: str) -> Path:
        return self._output_dir_for(dataset_name) / "samples.jsonl"

    def _results_path(self, dataset_name: str) -> Path:
        return self._output_dir_for(dataset_name) / "samples.jsonl_results.jsonl"

    def _results_txt_path(self, dataset_name: str) -> Path:
        return self._output_dir_for(dataset_name) / "results.txt"

    def _run_generation(self, dataset: Dict[str, Any]) -> bool:
        dataset_name = dataset["name"]
        data_file = Path(self.data_root) / dataset_name / dataset["data_file"]
        output_dir = self._output_dir_for(dataset_name)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_path = self._samples_path(dataset_name)

        problems = read_jsonl_file(str(data_file))
        print(f"  Total problems: {len(problems)}")

        # Resume from checkpoint: skip completed task_ids
        done_ids = []
        if save_path.exists():
            done_data = read_jsonl_file(str(save_path))
            done_ids = [x["task_id"] for x in done_data]
            print(f"  Skipping {len(done_ids)} completed tasks, {len(problems) - len(done_ids)} remaining")

        remaining = [p for p in problems if p["task_id"] not in done_ids]
        if not remaining:
            print(f"  All tasks completed")
            return True

        prompt_fn = construct_prompt_cot_text if self.prompt_variant == "cot" else construct_prompt_text

        with open(save_path, "a", encoding="utf-8") as f:
            for idx, problem in enumerate(remaining):
                print(f"[{idx+1}/{len(remaining)}] ", end="")
                task_id = problem["task_id"]
                prompt = prompt_fn(problem)

                try:
                    start_time = time.time()
                    responses, usage, model_name_resp, retry_time = get_response_text(prompt, self.api_config)
                    duration = time.time() - start_time

                    if responses and len(responses) > 0:
                        response_text = responses[0]
                        completion = extract_code(response_text)
                    else:
                        print(f"  [WARNING] Task {task_id} returned empty response")
                        response_text = "ERROR: Empty response"
                        completion = ""
                except Exception as e:
                    print(f"  [ERROR] Task {task_id} API failed: {e}")
                    import traceback
                    traceback.print_exc()
                    response_text = f"ERROR: {e}"
                    completion = ""
                    usage = {}
                    model_name_resp = self.model_name
                    duration = 0

                result = problem.copy()
                result["response"] = response_text
                result["completion"] = completion
                result["usage"] = usage if 'usage' in locals() else {}
                result["model"] = str(model_name_resp) if 'model_name_resp' in locals() else self.model_name
                result["duration"] = duration if 'duration' in locals() else 0

                try:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    print(f"  [SAVED] Task {task_id} completed, completion length: {len(completion)} chars")
                except Exception as e:
                    print(f"  [ERROR] Failed to save task {task_id}: {e}")

                # Request delay to prevent 429 rate limiting
                delay = self.api_config.get("request_delay", 1.0)
                if delay > 0:
                    time.sleep(delay)

        print(f"  Code generation completed: {save_path}")
        return True

    def _evaluate_lcb(self, samples_file: str, problem_file: str) -> float:
        """
        Evaluate LiveCodeBench samples using integrated LCB evaluation module.
        """
        print(f"  Running LCB evaluation...")
        
        # Use integrated LCB evaluation
        pass_at_1, results = evaluate_lcb_samples(samples_file, problem_file, timeout=6)
        
        # Save results to the same directory as samples
        output_dir = Path(samples_file).parent
        save_lcb_results(output_dir, pass_at_1, results)
        
        return pass_at_1

    def _run_evaluation(self, dataset: Dict[str, Any]) -> bool:
        dataset_name = dataset["name"]
        data_file = Path(self.data_root) / dataset_name / dataset["data_file"]
        samples_file = self._samples_path(dataset_name)

        if dataset_name == "LiveCodeBench":
            # Use LCB official evaluation
            print("Running LCB official evaluation...")
            # Use original LCB data file which contains test cases
            original_data_file = Path(self.data_root) / dataset_name / dataset.get("original_data_file", dataset["data_file"])
            pass_at_1 = self._evaluate_lcb(str(samples_file), str(original_data_file))
            print(f"  Pass@1: {pass_at_1:.4f}")
            return {"pass@1": pass_at_1}
        else:
            from human_eval.evaluation import evaluate_functional_correctness
            results = evaluate_functional_correctness(
                sample_file=str(samples_file),
                k=[1],
                n_workers=4,
                timeout=3.0,
                problem_file=str(data_file)
            )
            print(f"  Pass@1: {results.get('pass@1', 0):.4f}")
            return results

    def _calculate_token_stats(self, dataset_name: str) -> None:
        samples_file = self._samples_path(dataset_name)
        data_file = None
        for ds in self.DATASETS:
            if ds["name"] == dataset_name:
                data_file = Path(self.data_root) / dataset_name / ds["data_file"]
                break
        if data_file is None:
            data_file = Path(self.data_root) / dataset_name / f"{dataset_name}.jsonl"

        samples = read_jsonl_file(str(samples_file))

        # Group by difficulty
        easy, medium, hard = [], [], []
        for sample in samples:
            d = sample.get("difficulty", "").lower()
            if d == "easy":
                easy.append(sample)
            elif d == "medium":
                medium.append(sample)
            elif d == "hard":
                hard.append(sample)

        def avg_tokens(items):
            total, count = 0, 0
            for item in items:
                u = item.get("usage", {})
                if isinstance(u, dict) and u:
                    total += u.get("total_tokens", 0)
                    count += 1
            return total / count if count else 0

        token_stats = {
            "total_tasks": len(samples),
            "by_difficulty": {
                "easy": {"count": len(easy), "avg_tokens": avg_tokens(easy)},
                "medium": {"count": len(medium), "avg_tokens": avg_tokens(medium)},
                "hard": {"count": len(hard), "avg_tokens": avg_tokens(hard)},
            }
        }

        token_stats_path = self._output_dir_for(dataset_name) / "token_stats.json"
        with open(token_stats_path, 'w', encoding='utf-8') as f:
            json.dump(token_stats, f, indent=4, ensure_ascii=False)
        print(f"  Token statistics saved: {token_stats_path}")

    def _save_results_txt(self, dataset_name: str, pass_at_1: float) -> None:
        results_file = self._results_path(dataset_name)
        results_txt = self._results_txt_path(dataset_name)

        # For LCB, results_file might not exist, use samples_file instead
        if dataset_name == "LiveCodeBench" or not results_file.exists():
            results_file = self._samples_path(dataset_name)
        
        results = read_jsonl_file(str(results_file))

        # Calculate pass rate by difficulty
        easy, medium, hard = [], [], []
        for r in results:
            d = r.get("difficulty", "").lower()
            if d == "easy":
                easy.append(r)
            elif d == "medium":
                medium.append(r)
            elif d == "hard":
                hard.append(r)

        def pass_rate(items):
            if not items:
                return 0, 0, 0
            passed = sum(1 for x in items if x.get("passed"))
            return passed, len(items), passed / len(items)

        def token_stats_by_difficulty(items):
            total, count = 0, 0
            for item in items:
                u = item.get("usage", {})
                if isinstance(u, dict) and u:
                    total += u.get("total_tokens", 0)
                    count += 1
            return total / count if count else 0

        e_pass, e_n, e_r = pass_rate(easy)
        m_pass, m_n, m_r = pass_rate(medium)
        h_pass, h_n, h_r = pass_rate(hard)

        lines = [
            f"{{'pass@1': {pass_at_1}}}",
            "",
            f"Easy: {e_pass}/{e_n} = {e_r}",
            f"Medium: {m_pass}/{m_n} = {m_r}",
            f"Hard: {h_pass}/{h_n} = {h_r}",
            "",
            f"Easy Token Stats: Avg {token_stats_by_difficulty(easy):.2f} tokens/question",
            f"Medium Token Stats: Avg {token_stats_by_difficulty(medium):.2f} tokens/question",
            f"Hard Token Stats: Avg {token_stats_by_difficulty(hard):.2f} tokens/question",
        ]

        with open(results_txt, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))
        print(f"  Results saved: {results_txt}")

    def run_dataset(self, dataset: Dict[str, Any]) -> bool:
        dataset_name = dataset["name"]
        data_file = Path(self.data_root) / dataset_name / dataset["data_file"]

        print(f"\n{'='*70}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*70}")
        print(f"Data file: {data_file}")

        if not data_file.exists():
            print(f"[ERROR] Data file not found, skipping: {data_file}")
            return False

        data_hash = self._get_data_hash(dataset)

        # Check if already completed
        if self._check_completed(dataset_name, data_hash):
            print(f"[OK] Dataset {dataset_name} already completed, skipping")
            return True

        # Step 1: Code generation
        gen_ok = self._run_generation(dataset)
        if not gen_ok:
            self._update_progress(dataset_name, data_hash, "gen_failed", "Code generation failed")
            return False

        # Step 2: Evaluation
        if self._results_path(dataset_name).exists():
            print(f"  Evaluation results already exist, skipping evaluation")
        else:
            results = self._run_evaluation(dataset)
            pass_at_1 = results.get("pass@1", 0)

            # Token statistics
            self._calculate_token_stats(dataset_name)

            # Save results.txt
            self._save_results_txt(dataset_name, pass_at_1)

        self._update_progress(dataset_name, data_hash, "completed", "Evaluation completed")
        return True

    def run_all(self) -> Dict[str, Any]:
        print(f"{'='*70}")
        print(f"Text-to-Code Batch Evaluation")
        print(f"{'='*70}")
        print(f"Model: {self.model_name}")
        print(f"Data root: {self.data_root}")
        print(f"Output dir: {self.output_dir}")
        print(f"Datasets: {[d['name'] for d in self.DATASETS]}")
        print(f"Stream output: {'enabled' if self.enable_stream_output else 'disabled'}")

        results = {
            "model_name": self.model_name,
            "datasets": {},
            "summary": {"total": len(self.DATASETS), "completed": 0, "failed": 0}
        }

        for dataset in self.DATASETS:
            dataset_name = dataset["name"]
            try:
                ok = self.run_dataset(dataset)
                results["datasets"][dataset_name] = {
                    "status": "completed" if ok else "failed",
                    "message": "Completed" if ok else "Failed"
                }
                if ok:
                    results["summary"]["completed"] += 1
                else:
                    results["summary"]["failed"] += 1
            except Exception as e:
                print(f"[ERROR] Dataset {dataset_name} exception: {e}")
                import traceback
                traceback.print_exc()
                results["datasets"][dataset_name] = {"status": "failed", "message": str(e)}
                results["summary"]["failed"] += 1

        print(f"\n{'='*70}")
        print(f"Batch evaluation completed — Completed: {results['summary']['completed']}, Failed: {results['summary']['failed']}")

        summary_file = self.progress_file.parent / f"{self.model_name}_text_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Summary: {summary_file}")

        return results


def main():
    parser = argparse.ArgumentParser(description="Text-to-Code batch evaluation script (no image)")
    parser.add_argument("--api-config", type=str, required=True,
                        help="API config file path, e.g. src/configs/gemini_api_key_config.json")
    parser.add_argument("--data-root", type=str, default=None,
                        help=f"Data root directory (default: /home/zhangxu/CodeVision/data)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (default: ../output)")
    parser.add_argument("--prompt-variant", type=str, default="default",
                        choices=["default", "cot"],
                        help="Prompt type: default or cot (chain-of-thought)")
    parser.add_argument("--reset", action="store_true",
                        help="Reset progress and restart evaluation for all datasets")
    parser.add_argument("--stream-output", action="store_true",
                        help="Enable model stream output, print generation content in terminal in real-time")
    args = parser.parse_args()

    evaluator = UniversalBatchTextEvaluator(
        api_config=args.api_config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        prompt_variant=args.prompt_variant,
        enable_stream_output=args.stream_output
    )

    if args.reset and evaluator.progress_file.exists():
        evaluator.progress_file.unlink()
        print(f"Progress reset: {evaluator.progress_file}")

    results = evaluator.run_all()

    if results["summary"]["failed"] > 0:
        print("\n⚠ Some datasets failed, please check the logs")
        sys.exit(1)
    else:
        print("\n[OK] All dataset evaluations completed")
        sys.exit(0)


if __name__ == "__main__":
    main()
