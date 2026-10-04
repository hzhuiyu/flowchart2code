#!/usr/bin/env python3
"""
LCB Batch Reflection Agent
==========================
LiveCodeBench-specific batch reflection pipeline. Differences from the
standard batch_reflection_agent:

1. Instead of per-problem iteration, it runs a "global batch reflection +
   official evaluation" loop.
2. Each round:
   a. Reflect on all currently failed problems and regenerate code
   b. Batch-evaluate with the official LCB testing_util.run_test
   c. Only problems that still fail proceed to the next round
3. Up to 3 rounds; pass@1 statistics are printed after each round.

This script is a standalone module and does not modify the flowchart2code
core code.

Dependency:
    pip install numpy  # required by testing_util

Usage:
    # Step 0: generate initial results with flowchart2code
    cd flowchart2code/src && python main.py batch --dataset LiveCodeBench ...

    # Step 1: run LCB batch reflection
    cd flowchart2code
    python lcb_batch_reflection.py \
        --config src/configs/qwen_api_key_config.json \
        --samples output/LiveCodeBench/<model>/samples.jsonl \
        --lcb-data-dir ../LiveCodeBench/lcb_data \
        --release-version release_v1 \
        --max-rounds 3 \
        --reflection-mode code_only

    # Step 2: final evaluation (optional; the script already runs it)
    # Results are saved under output/LiveCodeBench/<model>/lcb_reflection/
"""

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Path setup: add flowchart2code/src and LiveCodeBench to PYTHONPATH
# ---------------------------------------------------------------------------
_FILE = Path(__file__).resolve()
_PROJECT_ROOT = _FILE.parent                     # flowchart2code/
_SRC_DIR = _PROJECT_ROOT / "src"                 # flowchart2code/src/
_LCB_ROOT = _PROJECT_ROOT.parent / "LiveCodeBench"  # LiveCodeBench/

for _p in (_SRC_DIR, _LCB_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# ---------------------------------------------------------------------------
# Deferred imports (avoid import errors before the paths above are set)
# ---------------------------------------------------------------------------
# flowchart2code core components
from utils.vision_api_client import VisionAPIClient
from agents.agent2_code_generator import CodeGenerator
from utils.code_utils import extract_code

# Official LCB evaluation
from lcb_runner.evaluation.testing_util import run_test
from lcb_local_loader import load_lcb_local, CodeGenerationProblem


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LCBFailedSample:
    """LCB failed sample (trimmed to the fields needed for reflection)."""
    task_id: str
    question_id: str
    original_code: str
    image_path: str
    starter_code: str = ""
    prompt: str = ""
    difficulty: str = "easy"
    # Used by the code generator
    problem_text_description: str = ""
    entry_point: str = ""
    test_cases: List[Dict] = field(default_factory=list)


@dataclass
class RoundResult:
    """Result of one reflection + evaluation round."""
    round_num: int
    task_id: str
    new_code: str
    lcb_passed: bool          # whether the official LCB evaluation passed
    lcb_results: List[Any]    # results list returned by run_test
    metadata: Dict[str, Any]  # metadata returned by run_test
    analysis: str = ""        # reflection analysis text
    token_usage: Dict = field(default_factory=dict)
    duration: float = 0.0


# ---------------------------------------------------------------------------
# LCB evaluation wrapper
# ---------------------------------------------------------------------------

def lcb_official_evaluate(
    code: str,
    problem: CodeGenerationProblem,
    timeout: int = 6,
) -> Tuple[bool, List[Any], Dict[str, Any]]:
    """
    Evaluate a single problem's code with the official LCB run_test.

    Returns:
        (passed_all, results_list, metadata)
        passed_all: whether all test cases passed
        results_list: per-test-case results
        metadata: evaluation metadata
    """
    sample = problem.get_evaluation_sample()
    sample["code"] = code  # compatibility with some variants

    results, metadata = run_test(
        sample=sample,
        test=code,
        debug=False,
        timeout=timeout,
    )

    # results format: each element is True (pass), False (fail), -2 (WA),
    # -3 (TLE), -4 (RE)
    if not isinstance(results, list):
        # unexpected case
        return False, [], metadata

    passed_all = all(r is True for r in results)
    return passed_all, results, metadata


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class LCBBatchReflectionAgent:
    """
    LiveCodeBench-specific batch reflection agent.

    Key differences (vs the standard BatchReflectionAgent):
    - Each round batch-reflects on all failed problems first, then evaluates
      them uniformly with the official LCB runner
    - No ILR logic; only code-level reflection (code_only mode is sufficient)
    - Up to 3 rounds; global pass@1 is printed after each round
    """

    def __init__(
        self,
        config_path: str,
        samples_file: str,
        lcb_data_dir: str,
        release_version: str = "release_v1",
        output_dir: Optional[str] = None,
        max_rounds: int = 3,
        reflection_mode: str = "code_only",
        prompt_variant: str = "default",
        include_problem_text_description: bool = True,
        eval_timeout: int = 6,
    ):
        """
        Args:
            config_path: LLM config file path
            samples_file: samples.jsonl path produced by flowchart2code
            lcb_data_dir: LiveCodeBench data directory (containing test.jsonl)
            release_version: LCB version, e.g. release_v1
            output_dir: output directory; defaults to lcb_reflection/ next to
                samples_file
            max_rounds: max reflection rounds (default 3)
            reflection_mode: only code_only is supported (LCB needs no ILR)
            eval_timeout: official LCB evaluation timeout (seconds)
        """
        # Load config
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)

        self.api_client = VisionAPIClient(config_dict=self.config)
        self.code_generator = CodeGenerator(
            config_dict=self.config,
            output_dir=str(output_dir) if output_dir else None,
            prompt_variant=prompt_variant,
            include_problem_text_description=include_problem_text_description,
        )

        self.samples_file = Path(samples_file)
        self.lcb_data_dir = Path(lcb_data_dir)
        self.release_version = release_version
        self.max_rounds = max_rounds
        self.eval_timeout = eval_timeout
        self.prompt_variant = prompt_variant
        self.include_problem_text_description = include_problem_text_description

        # Output directory
        if output_dir is None:
            self.output_dir = self.samples_file.parent / "lcb_reflection"
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load official LCB problems (for test cases and evaluation)
        print(f"Loading LCB problems from {lcb_data_dir} ({release_version})...")
        self.lcb_problems: Dict[str, CodeGenerationProblem] = {}
        for prob in load_lcb_local(
            data_dir=lcb_data_dir,
            release_version=release_version,
        ):
            self.lcb_problems[prob.question_id] = prob
        print(f"Loaded {len(self.lcb_problems)} LCB problems")

        # Load current samples
        self.all_samples: Dict[str, Dict[str, Any]] = {}
        self._load_samples()

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def _load_samples(self):
        """Read the samples.jsonl produced by flowchart2code."""
        source_items = {}
        with open(self.samples_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                    tid = item.get("task_id")
                    if tid:
                        source_items[tid] = item
                except json.JSONDecodeError:
                    continue
        # Evaluator result files do not include completions; recover them from
        # the sibling samples.jsonl and merge pass/fail metadata by task_id.
        if source_items and not any(item.get("completion") for item in source_items.values()):
            sibling = self.samples_file.with_name("samples.jsonl")
            if sibling.exists() and sibling != self.samples_file:
                with sibling.open("r", encoding="utf-8") as f:
                    for line in f:
                        try:
                            item = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        tid = item.get("task_id")
                        if tid and tid in source_items:
                            merged = dict(item)
                            merged.update({
                                key: value
                                for key, value in source_items[tid].items()
                                if key in {"passed", "test_results", "metadata"}
                            })
                            source_items[tid] = merged
        self.all_samples = source_items
        print(f"Loaded {len(self.all_samples)} samples from {self.samples_file}")

    def _build_failed_samples(
        self, failed_task_ids: List[str]
    ) -> List[LCBFailedSample]:
        """Build LCBFailedSample objects from the failed task-id list."""
        failed = []
        for tid in failed_task_ids:
            sample = self.all_samples.get(tid)
            if not sample:
                continue

            # Parse response for extra information
            response = sample.get("response", "{}")
            if isinstance(response, str):
                try:
                    resp = json.loads(response)
                except json.JSONDecodeError:
                    resp = {}
            else:
                resp = response

            problem_info = resp.get("problem_info", {}) if resp else {}

            # Locate the image path
            image_path = self._find_image_path(tid)
            if not image_path:
                print(f"Warning: image not found for {tid}, skipping reflection")
                continue

            # starter_code / prompt
            starter_code = (
                problem_info.get("starter_code")
                or sample.get("starter_code", "")
            )
            prompt_text = sample.get("prompt", problem_info.get("title", ""))
            difficulty = sample.get("difficulty", problem_info.get("difficulty", "easy"))
            text_desc = problem_info.get("text_description", "")
            entry_point = problem_info.get("entry_point", "")
            test_cases = problem_info.get("test_cases", [])

            code = sample.get("completion", "")
            if not code and resp:
                code = resp.get("generated_code", resp.get("completion", ""))
            code = extract_code(code) if isinstance(code, str) else str(code)

            failed.append(
                LCBFailedSample(
                    task_id=tid,
                    question_id=tid,
                    original_code=code,
                    image_path=str(image_path),
                    starter_code=starter_code,
                    prompt=prompt_text,
                    difficulty=difficulty,
                    problem_text_description=text_desc,
                    entry_point=entry_point,
                    test_cases=test_cases,
                )
            )
        return failed

    def _find_image_path(self, task_id: str) -> Optional[Path]:
        """Look up the flowchart image under data/LiveCodeBench/images."""
        img_dir = _PROJECT_ROOT / "data" / "LiveCodeBench" / "images"
        for ext in (".png", ".jpg", ".jpeg"):
            p = img_dir / f"{task_id}{ext}"
            if p.exists():
                return p
        return None

    # ------------------------------------------------------------------
    # Per-problem reflection (reuses the existing prompt logic, code only)
    # ------------------------------------------------------------------

    def _reflect_code(self, sample: LCBFailedSample) -> Tuple[str, str, Dict]:
        """
        Reflect on a single failed sample; returns (new_code, analysis, usage).
        """
        # Build problem_info for the CodeGenerator
        problem_info = {
            "entry_point": sample.entry_point,
            "test_cases": sample.test_cases,
            "starter_code": sample.starter_code,
            "text_description": sample.problem_text_description,
            "prompt": sample.prompt,
        }

        # Use the CodeGenerator's code reflection method
        # Note: use generate_code_from_code_reflection if available,
        # otherwise fall back to plain generate_code + custom prompt
        try:
            new_code, usage = self.code_generator.generate_code_from_code_reflection(
                problem_info=problem_info,
                original_code=sample.original_code,
                analysis="",  # let the LLM analyze on its own first
                image_path=sample.image_path,
            )
            analysis = "code_reflection_via_generator"
        except AttributeError:
            # If CodeGenerator has no reflection method, fall back to
            # direct regeneration
            print(f"  > CodeGenerator has no reflection method, regenerating normally...")
            # Build a simple prompt so the LLM improves on the original code
            analysis = self._get_code_analysis(sample)
            new_code, usage = self._generate_code_with_analysis(sample, analysis)

        # Clean the code
        new_code = extract_code(new_code) if isinstance(new_code, str) else str(new_code)
        return new_code, analysis, usage or {}

    def _get_code_analysis(self, sample: LCBFailedSample) -> str:
        """Ask the LLM why the code failed (simplified prompt)."""
        prompt = f"""You are an expert Python programmer.
The following code failed some test cases. Please analyze the code and identify bugs.

Task ID: {sample.task_id}
Problem: {sample.prompt}

Current Code:
```python
{sample.original_code}
```

Starter Code / Interface:
```python
{sample.starter_code}
```

Please provide a concise analysis (max 300 words):
1. Brief diagnosis of the bug
2. Specific fix suggestions
Do NOT generate the full corrected code yet.
"""
        try:
            result = self.api_client.call_api(prompt=prompt, image_path=None)
            return result.get("content", "")
        except Exception as e:
            print(f"  > Analysis API call failed: {e}")
            return f"Auto-analysis failed: {e}"

    def _generate_code_with_analysis(
        self, sample: LCBFailedSample, analysis: str
    ) -> Tuple[str, Dict]:
        """Regenerate code based on the analysis."""
        prompt = f"""You are an expert Python programmer.
Based on the analysis below, regenerate the corrected code.

Task ID: {sample.task_id}
Problem: {sample.prompt}

Starter Code / Interface:
```python
{sample.starter_code}
```

Analysis of bugs:
{analysis}

Original Code:
```python
{sample.original_code}
```

Please output ONLY the corrected Python code. Do NOT include explanations.
Make sure the code follows the required interface (class/function names from starter code).
"""
        try:
            result = self.api_client.call_api(prompt=prompt, image_path=None)
            code = extract_code(result.get("content", ""))
            return code, result.get("usage", {})
        except Exception as e:
            print(f"  > Code regeneration failed: {e}")
            return sample.original_code, {}

    # ------------------------------------------------------------------
    # Batch evaluation
    # ------------------------------------------------------------------

    def _evaluate_all_with_lcb(
        self, task_ids: List[str], code_map: Dict[str, str]
    ) -> Dict[str, Tuple[bool, List[Any], Dict]]:
        """
        Evaluate multiple problems with the official LCB run_test.
        Returns: {task_id: (passed_all, results_list, metadata)}
        """
        eval_results = {}
        total = len(task_ids)
        for idx, tid in enumerate(task_ids, 1):
            prob = self.lcb_problems.get(tid)
            code = code_map.get(tid, "")
            if not prob or not code:
                eval_results[tid] = (False, [], {"error": "missing problem or code"})
                continue

            print(f"  [{idx}/{total}] Evaluating {tid} ...", end=" ")
            try:
                passed, results, meta = lcb_official_evaluate(
                    code=code,
                    problem=prob,
                    timeout=self.eval_timeout,
                )
                status = "PASS" if passed else "FAIL"
                print(f"{status}")
                eval_results[tid] = (passed, results, meta)
            except Exception as e:
                print(f"ERROR: {e}")
                eval_results[tid] = (False, [], {"error": str(e)})
        return eval_results

    # ------------------------------------------------------------------
    # Single round: reflect -> evaluate
    # ------------------------------------------------------------------

    def run_single_round(
        self,
        round_num: int,
        failed_task_ids: List[str],
    ) -> Tuple[List[str], Dict[str, RoundResult]]:
        """
        Run one "batch reflection + official evaluation" round.

        Returns:
            (still_failed_ids, round_results_map)
        """
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{self.max_rounds}: {len(failed_task_ids)} failed samples")
        print(f"{'='*60}")

        # 1. Build failed samples
        failed_samples = self._build_failed_samples(failed_task_ids)
        if not failed_samples:
            return [], {}

        # 2. Batch reflection: generate new code for each failed sample
        new_code_map: Dict[str, str] = {}
        round_results: Dict[str, RoundResult] = {}

        for idx, sample in enumerate(failed_samples, 1):
            print(f"\n[{idx}/{len(failed_samples)}] Reflecting {sample.task_id} ...")
            start_time = time.time()

            new_code, analysis, usage = self._reflect_code(sample)
            duration = time.time() - start_time

            new_code_map[sample.task_id] = new_code
            round_results[sample.task_id] = RoundResult(
                round_num=round_num,
                task_id=sample.task_id,
                new_code=new_code,
                lcb_passed=False,  # updated after evaluation
                lcb_results=[],
                metadata={},
                analysis=analysis,
                token_usage=usage,
                duration=duration,
            )

            # Save this round's generated code (for later merging)
            self._save_round_code(round_num, sample.task_id, new_code)

        # 3. Batch-test with the official LCB evaluation
        print(f"\n--- Running LCB official evaluation for round {round_num} ---")
        eval_results = self._evaluate_all_with_lcb(failed_task_ids, new_code_map)

        # 4. Update results and collect still-failed problems
        still_failed = []
        for tid in failed_task_ids:
            passed, results, meta = eval_results.get(tid, (False, [], {}))
            rr = round_results.get(tid)
            if rr:
                rr.lcb_passed = passed
                rr.lcb_results = results
                rr.metadata = meta

            if passed:
                print(f"  ✓ {tid} PASSED in round {round_num}")
                # Mark the global sample as passed
                self._update_sample_as_passed(tid, new_code_map.get(tid, ""))
            else:
                still_failed.append(tid)
                # Keep the latest code on the global sample (for the next
                # round) even though it failed
                self._update_sample_code(tid, new_code_map.get(tid, ""))

        # 5. Save this round's detailed results
        self._save_round_results(round_num, round_results)

        return still_failed, round_results

    # ------------------------------------------------------------------
    # Result saving and merging
    # ------------------------------------------------------------------

    def _save_round_code(self, round_num: int, task_id: str, code: str):
        """Save the code generated in one round."""
        code_dir = self.output_dir / f"round_{round_num}_codes"
        code_dir.mkdir(exist_ok=True)
        with open(code_dir / f"{task_id}.py", "w", encoding="utf-8") as f:
            f.write(code)

    def _save_round_results(self, round_num: int, round_results: Dict[str, RoundResult]):
        """Save one round's evaluation results to JSONL."""
        result_file = self.output_dir / f"round_{round_num}_results.jsonl"
        with open(result_file, "w", encoding="utf-8") as f:
            for tid, rr in round_results.items():
                record = {
                    "task_id": tid,
                    "round": rr.round_num,
                    "passed": rr.lcb_passed,
                    "results": rr.lcb_results,
                    "metadata": rr.metadata,
                    "analysis": rr.analysis,
                    "token_usage": rr.token_usage,
                    "duration": rr.duration,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _update_sample_as_passed(self, task_id: str, code: str):
        """Mark a sample as passed (update memory and file)."""
        if task_id in self.all_samples:
            self.all_samples[task_id]["completion"] = code
            self.all_samples[task_id]["lcb_passed"] = True

    def _update_sample_code(self, task_id: str, code: str):
        """Update a sample's code (for the next round)."""
        if task_id in self.all_samples:
            self.all_samples[task_id]["completion"] = code
            self.all_samples[task_id]["lcb_passed"] = False

    def _save_final_samples(self):
        """Save the final samples.jsonl (latest code after all rounds)."""
        final_file = self.output_dir / "samples_final.jsonl"
        with open(final_file, "w", encoding="utf-8") as f:
            for tid in sorted(self.all_samples.keys()):
                f.write(json.dumps(self.all_samples[tid], ensure_ascii=False) + "\n")
        print(f"\nFinal samples saved to: {final_file}")
        return final_file

    def _compute_pass_at_1(self, passed_ids: set) -> float:
        """Compute pass@1."""
        total = len(self.all_samples)
        if total == 0:
            return 0.0
        return len(passed_ids) / total

    # ------------------------------------------------------------------
    # Main flow
    # ------------------------------------------------------------------

    def run(self) -> Dict[str, Any]:
        """
        Run the complete LCB batch reflection pipeline.

        Returns:
            Final statistics
        """
        # Initial evaluation: determine which problems already pass
        print("\n" + "="*60)
        print("Initial LCB Evaluation")
        print("="*60)
        initial_codes = {tid: s.get("completion", "") for tid, s in self.all_samples.items()}
        initial_eval = self._evaluate_all_with_lcb(list(self.all_samples.keys()), initial_codes)

        passed_ids = set()
        failed_ids = []
        for tid, (passed, results, meta) in initial_eval.items():
            if passed:
                passed_ids.add(tid)
                self.all_samples[tid]["lcb_passed"] = True
            else:
                failed_ids.append(tid)
                self.all_samples[tid]["lcb_passed"] = False

        print(f"\nInitial: {len(passed_ids)}/{len(self.all_samples)} passed ({self._compute_pass_at_1(passed_ids)*100:.2f}%)")

        # Multi-round reflection
        all_round_summaries = []
        for round_num in range(1, self.max_rounds + 1):
            if not failed_ids:
                print(f"\nAll samples passed! Stopping at round {round_num - 1}.")
                break

            still_failed, round_results = self.run_single_round(round_num, failed_ids)

            # Round statistics
            newly_passed = set(failed_ids) - set(still_failed)
            passed_ids.update(newly_passed)
            pass_at_1 = self._compute_pass_at_1(passed_ids)

            summary = {
                "round": round_num,
                "processed": len(failed_ids),
                "newly_passed": len(newly_passed),
                "still_failed": len(still_failed),
                "total_passed": len(passed_ids),
                "total_samples": len(self.all_samples),
                "pass_at_1": pass_at_1,
            }
            all_round_summaries.append(summary)

            print(f"\nRound {round_num} Summary:")
            print(f"  Processed: {summary['processed']}")
            print(f"  Newly Passed: {summary['newly_passed']}")
            print(f"  Still Failed: {summary['still_failed']}")
            print(f"  Total Passed: {summary['total_passed']}/{summary['total_samples']}")
            print(f"  Pass@1: {pass_at_1*100:.2f}%")

            failed_ids = still_failed

        # Save final results
        final_file = self._save_final_samples()

        # Overall statistics
        final_summary = {
            "initial_passed": len([t for t, s in self.all_samples.items() if s.get("lcb_passed")]),
            "total_samples": len(self.all_samples),
            "final_pass_at_1": self._compute_pass_at_1(
                {t for t, s in self.all_samples.items() if s.get("lcb_passed")}
            ),
            "rounds": all_round_summaries,
            "output_file": str(final_file),
        }

        # Save summary JSON
        summary_file = self.output_dir / "lcb_reflection_summary.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(final_summary, f, ensure_ascii=False, indent=2)

        print("\n" + "="*60)
        print("LCB Batch Reflection Complete!")
        print("="*60)
        print(f"Final Pass@1: {final_summary['final_pass_at_1']*100:.2f}%")
        print(f"Summary saved to: {summary_file}")
        print(f"Final samples saved to: {final_file}")

        return final_summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LCB Batch Reflection Agent")
    parser.add_argument("--config", required=True, help="LLM config JSON path")
    parser.add_argument("--samples", required=True, help="flowchart2code samples.jsonl path")
    parser.add_argument("--lcb-data-dir", required=True, help="LCB data directory (with test.jsonl)")
    parser.add_argument("--release-version", default="release_v1", help="LCB release version")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--max-rounds", type=int, default=3, help="Max reflection rounds")
    parser.add_argument("--reflection-mode", default="code_only", help="Only code_only supported for now")
    parser.add_argument("--prompt-variant", default="default", choices=["default", "self_planning"])
    parser.add_argument("--eval-timeout", type=int, default=6, help="LCB evaluation timeout per test")
    parser.add_argument("--include-problem-text-description", type=lambda x: x.lower() == "true", default=True)

    args = parser.parse_args()

    agent = LCBBatchReflectionAgent(
        config_path=args.config,
        samples_file=args.samples,
        lcb_data_dir=args.lcb_data_dir,
        release_version=args.release_version,
        output_dir=args.output_dir,
        max_rounds=args.max_rounds,
        reflection_mode=args.reflection_mode,
        prompt_variant=args.prompt_variant,
        include_problem_text_description=args.include_problem_text_description,
        eval_timeout=args.eval_timeout,
    )

    agent.run()


if __name__ == "__main__":
    main()
