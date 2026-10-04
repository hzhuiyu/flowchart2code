#!/usr/bin/env python3
"""
Batch evaluation script - LangGraph multi-agent workflow
Supports running three datasets based on the provided config, evaluating and saving results after each dataset completes
Supports checkpoint resume, does not depend on task_id for progress tracking
"""

import sys
import os
import json
import argparse
import time
import io
import contextlib
from pathlib import Path
from typing import Dict, List, Any
import hashlib
from utils.intermediate_representation import (
    get_intermediate_representation_suffix,
    normalize_intermediate_representation_type,
)

# Keep batch progress logging portable on Windows consoles using a legacy code page.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def str2bool(value):
    """Parse command line boolean argument, supports true/false."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean value: {value}")


class ResumableBatchEvaluator:
    """Resumable batch evaluator with checkpoint support"""
    
    def __init__(self, vision_model_2_config: str, language_model_1_config: str,
                 data_root: str = None, output_dir: str = None,
                 reflection_model_config: str = None, use_reflection: bool = False,
                 max_iterations: int = 3,
                 intermediate_representation_type: str = "ilr",
                 prompt_variant: str = "default",
                 dataset_names: List[str] = None,
                 enable_stream_output: bool = False,
                 include_problem_text_description: bool = True):
        """
        Initialize the evaluator

        Args:
            vision_model_2_config: Vision model 2 config file path (for pseudocode generation)
            language_model_1_config: Language model 1 config file path (for code generation)
            data_root: Data root directory
            output_dir: Output directory
            reflection_model_config: Reflection model config file path (optional)
            use_reflection: Whether to use reflection
            max_iterations: Maximum iteration count (default 3)
        """
        from utils.vision_api_client import VisionAPIClient

        # Load configuration files
        self.vision_model_2_config = self._load_config(vision_model_2_config)
        self.language_model_1_config = self._load_config(language_model_1_config)
        self.use_reflection = use_reflection
        self.max_iterations = max_iterations
        self.intermediate_representation_type = normalize_intermediate_representation_type(
            intermediate_representation_type
        )
        self.prompt_variant = (prompt_variant or "default").strip().lower()
        self.enable_stream_output = enable_stream_output
        self.include_problem_text_description = include_problem_text_description

        # If using reflection, load the reflection model config
        if use_reflection:
            if reflection_model_config is None:
                raise ValueError("Must provide reflection_model_config when using reflection functionality")
            self.reflection_model_config = self._load_config(reflection_model_config)
        else:
            self.reflection_model_config = None

        if self.enable_stream_output:
            self._apply_stream_settings(self.vision_model_2_config, "vision")
            self._apply_stream_settings(self.language_model_1_config, "language")
            if self.reflection_model_config:
                self._apply_stream_settings(self.reflection_model_config, "reflection")
        
        # Set paths
        if data_root is None:
            data_root = str(Path(__file__).parent.parent / "data")
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent / "output")
        
        self.data_root = data_root
        self.output_dir = output_dir
        
        # Get model name (for output folder)
        self.model_name = self._get_model_name()
        
        # Define dataset list
        all_datasets = [
            {
                "name": "HumanEval-V",
                "data_file": "HumanEval.jsonl",
                "images_dir": "images"
            },
            {
                "name": "Algorithm",
                "data_file": "Algorithm.jsonl",
                "images_dir": "images"
            },
            {
                "name": "MATH",
                "data_file": "MATH.jsonl",
                "images_dir": "images"
            },
            {
                "name": "LiveCodeBench",
                "data_file": "LiveCodeBench.jsonl",
                "original_data_file": "LiveCodeBench_merged.jsonl",
                "images_dir": "images"
            }
        ]

        if dataset_names:
            requested = set(dataset_names)
            available = {dataset["name"] for dataset in all_datasets}
            invalid = sorted(requested - available)
            if invalid:
                raise ValueError(
                    f"Unsupported dataset(s): {', '.join(invalid)}. Available datasets: {', '.join(sorted(available))}"
                )
            self.datasets = [
                dataset for dataset in all_datasets
                if dataset["name"] in requested
            ]
        else:
            self.datasets = all_datasets
        
        # Progress file path
        self.progress_file = Path(self.output_dir) / "progress" / f"{self.model_name}_progress.json"
        self.progress_file.parent.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration file"""
        if not Path(config_path).exists():
            # Try to find in the configs directory
            config_path = Path(__file__).parent / "configs" / config_path
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _get_model_name(self) -> str:
        """Get the output folder name"""
        import re

        def clean_name(name: str) -> str:
            name = re.sub(r'[^a-zA-Z0-9-]', '-', name)
            return re.sub(r'-+', '-', name).strip('-')

        vision_2 = clean_name(self.vision_model_2_config.get("model", "vision"))
        language = clean_name(self.language_model_1_config.get("model", "language"))

        # If using reflection, add the reflection model name
        suffix = get_intermediate_representation_suffix(self.intermediate_representation_type)

        if self.use_reflection and self.reflection_model_config:
            reflection = clean_name(self.reflection_model_config.get("model", "reflection"))
            name = f"{vision_2}-{language}-{reflection}{suffix}"
        else:
            name = f"{vision_2}-{language}{suffix}"
        if self.prompt_variant == "self_planning":
            name += "_self_planning"
        return name

    def _apply_stream_settings(self, config: Dict[str, Any], role: str) -> None:
        """Inject streaming output options into model configuration"""
        config["stream"] = True
        config.setdefault("stream_label", f"{role}:{config.get('model', 'unknown-model')}")

    def _write_captured_log(self, dataset_name: str, stage_name: str, content: str) -> Path:
        """Save detailed logs captured in silent mode."""
        logs_dir = Path(self.output_dir) / "logs" / self.model_name
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_file = logs_dir / f"{dataset_name}_{stage_name}.log"
        log_file.write_text(content, encoding='utf-8')
        return log_file

    def _run_with_optional_output_capture(self, dataset_name: str, stage_name: str, callback):
        """
        Suppress subprocess output in non-streaming mode to avoid flooding
        the terminal with per-sample logs during batch processing.

        Returns:
            (result, captured_output)
        """
        if self.enable_stream_output:
            return callback(), ""

        stdout_buffer = io.StringIO()

        try:
            with contextlib.redirect_stdout(stdout_buffer):
                result = callback()
        except Exception:
            captured_output = stdout_buffer.getvalue()
            if captured_output.strip():
                log_file = self._write_captured_log(dataset_name, stage_name, captured_output)
                print(f"  ⚠ {stage_name} stage detailed logs saved to: {log_file}")
            raise

        captured_output = stdout_buffer.getvalue()
        return result, captured_output
    
    def _load_progress(self) -> Dict[str, Any]:
        """Load progress file"""
        if not self.progress_file.exists():
            return {
                "model_name": self.model_name,
                "datasets": {},
                "last_update": None
            }
        
        with open(self.progress_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def _save_progress(self, progress: Dict[str, Any]) -> None:
        """Save progress file"""
        progress["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        with open(self.progress_file, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2, ensure_ascii=False)
    
    def _get_dataset_hash(self, dataset: Dict[str, Any]) -> str:
        """Compute the hash of the dataset to detect whether it has changed"""
        data_file = Path(self.data_root) / dataset["name"] / dataset["data_file"]
        images_dir = Path(self.data_root) / dataset["name"] / dataset["images_dir"]
        
        if not data_file.exists() or not images_dir.exists():
            return None
        
        # Compute the hash of the data file
        with open(data_file, 'rb') as f:
            data_hash = hashlib.md5(f.read()).hexdigest()
        
        # Compute the hash of the images directory (based on file list and modification time)
        image_files = sorted(images_dir.glob("*.png"))
        image_info = []
        for img_file in image_files:
            stat = img_file.stat()
            image_info.append(f"{img_file.name}:{stat.st_size}:{stat.st_mtime}")
        image_hash = hashlib.md5("|".join(image_info).encode()).hexdigest()
        
        return f"{data_hash}:{image_hash}"
    
    def _check_dataset_completed(self, dataset_name: str, dataset_hash: str) -> bool:
        """Check whether the dataset has been completed"""
        progress = self._load_progress()
        
        if dataset_name not in progress["datasets"]:
            return False
        
        dataset_progress = progress["datasets"][dataset_name]
        
        # Check if the hash matches
        if dataset_progress.get("hash") != dataset_hash:
            print(f"  ⚠ Dataset {dataset_name} has changed, needs re-evaluation")
            return False
        
        # Check if marked as completed
        return dataset_progress.get("completed", False)
    
    def _update_dataset_progress(self, dataset_name: str, dataset_hash: str,
                                  status: str, message: str = None) -> None:
        """Update dataset progress"""
        progress = self._load_progress()
        
        if dataset_name not in progress["datasets"]:
            progress["datasets"][dataset_name] = {
                "hash": dataset_hash,
                "status": status,
                "completed": False,
                "messages": []
            }
        
        dataset_progress = progress["datasets"][dataset_name]
        dataset_progress["hash"] = dataset_hash
        dataset_progress["status"] = status
        dataset_progress["last_update"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if status == "completed":
            dataset_progress["completed"] = True
        
        if message:
            dataset_progress["messages"].append({
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "message": message
            })
        
        self._save_progress(progress)
    
    def _check_samples_completed(self, dataset_name: str, total_images: int) -> bool:
        """Check whether samples.jsonl is complete"""
        samples_file = Path(self.output_dir) / dataset_name / self.model_name / "samples.jsonl"
        
        if not samples_file.exists():
            return False
        
        # Read samples.jsonl
        samples = []
        with open(samples_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    samples.append(json.loads(line.strip()))
                except:
                    continue
        
        # Check sample count
        if len(samples) < total_images:
            return False
        
        # Check for errors
        for sample in samples:
            response = json.loads(sample.get("response", "{}"))
            if response.get("status") == "failed":
                return False
        
        return True
    
    def _check_evaluation_completed(self, dataset_name: str) -> bool:
        """Check whether the evaluation is complete"""
        samples_file = Path(self.output_dir) / dataset_name / self.model_name / "samples.jsonl"
        results_file = Path(self.output_dir) / dataset_name / self.model_name / "samples.jsonl_results.jsonl"
        results_txt = Path(self.output_dir) / dataset_name / self.model_name / "results.txt"
        
        if not samples_file.exists():
            return False
        
        if not results_file.exists():
            return False
        
        if not results_txt.exists():
            return False
        
        return True
    
    def run_dataset(self, dataset: Dict[str, Any]) -> bool:
        """Run evaluation for a single dataset"""
        dataset_name = dataset["name"]
        data_file = Path(self.data_root) / dataset_name / dataset["data_file"]
        images_dir = Path(self.data_root) / dataset_name / dataset["images_dir"]
        
        print(f"\n{'='*70}")
        print(f"Starting evaluation for dataset: {dataset_name}")
        print(f"{'='*70}")
        print(f"Data file: {data_file}")
        print(f"Images directory: {images_dir}")
        
        # Compute dataset hash
        dataset_hash = self._get_dataset_hash(dataset)
        if dataset_hash is None:
            print(f"✗ Dataset {dataset_name} files not found, skipping")
            return False
        
        # Check if already completed
        if self._check_dataset_completed(dataset_name, dataset_hash):
            print(f"✓ Dataset {dataset_name} already completed, skipping")
            return True
        
        # Get image count
        image_files = list(images_dir.glob("*.png"))
        total_images = len(image_files)
        
        if total_images == 0:
            print(f"✗ Dataset {dataset_name} has no images, skipping")
            return False
        
        print(f"Total images: {total_images}")
        
        # Check if samples.jsonl is complete
        samples_completed = self._check_samples_completed(dataset_name, total_images)
        if samples_completed:
            print(f"✓ samples.jsonl already complete ({total_images} samples)")
        else:
            print(f"⏳ Starting code generation...")
            # Generate code
            try:
                from main import FlowchartProcessor

                processor = FlowchartProcessor(
                    data_root=self.data_root,
                    vision_model_2_config=self.vision_model_2_config,
                    language_model_1_config=self.language_model_1_config,
                    reflection_model_config=self.reflection_model_config,
                    use_reflection=self.use_reflection,
                    max_iterations=self.max_iterations,
                    intermediate_representation_type=self.intermediate_representation_type,
                    prompt_variant=self.prompt_variant,
                    include_problem_text_description=self.include_problem_text_description,
                    output_dir=self.output_dir,
                    enable_evaluation=False  # Don't evaluate yet; evaluate after all samples are generated
                )

                summary, generation_logs = self._run_with_optional_output_capture(
                    dataset_name,
                    "generation",
                    lambda: processor.process_dataset_batch(
                        dataset=dataset_name,
                        limit=None,
                        use_langgraph=True,
                        enable_evaluation=False
                    ),
                )

                print(f"✓ Code generation complete: {summary['success']}/{summary['total']} succeeded")
                if generation_logs.strip() and summary.get("failed", 0) > 0:
                    log_file = self._write_captured_log(dataset_name, "generation", generation_logs)
                    print(f"  ⚠ Generation stage contains failed samples, detailed logs: {log_file}")
                
                # Update progress
                self._update_dataset_progress(
                    dataset_name, dataset_hash,
                    "samples_completed",
                    f"Code generation complete: {summary['success']}/{summary['total']} succeeded"
                )
                
            except Exception as e:
                print(f"✗ Code generation failed: {e}")
                self._update_dataset_progress(
                    dataset_name, dataset_hash,
                    "samples_failed",
                    f"Code generation failed: {e}"
                )
                return False
        
        # Check if evaluation is complete
        evaluation_completed = self._check_evaluation_completed(dataset_name)
        if evaluation_completed:
            print(f"✓ Evaluation already complete")
        else:
            print(f"⏳ Starting evaluation...")
            # Evaluate code
            try:
                if dataset_name == "LiveCodeBench":
                    # Use the LCB mixed I/O evaluator instead of the HumanEval interface.
                    from evaluation.evaluate_lcb import evaluate_lcb_samples, save_lcb_results

                    samples_file = Path(self.output_dir) / dataset_name / self.model_name / "samples.jsonl"
                    problem_file = (
                        Path(self.data_root) / dataset_name /
                        dataset.get("original_data_file", "LiveCodeBench_merged.jsonl")
                    )
                    pass_at_1, lcb_results = evaluate_lcb_samples(
                        str(samples_file), str(problem_file), timeout=6
                    )
                    save_lcb_results(samples_file.parent, pass_at_1, lcb_results)
                    print(f"✓ LCB evaluation complete")
                    print(f"  Pass@1: {pass_at_1:.4f}")
                    self._update_dataset_progress(
                        dataset_name, dataset_hash,
                        "completed",
                        f"LCB evaluation complete: Pass@1 = {pass_at_1:.4f}"
                    )
                    return True

                # Choose different agents based on whether reflection is used
                if self.use_reflection:
                    from agents.langgraph_agent_with_rethink import LangGraphFlowchartAgentWithRethink

                    agent = LangGraphFlowchartAgentWithRethink(
                        data_root=self.data_root,
                        vision_model_2_config=self.vision_model_2_config,
                        language_model_1_config=self.language_model_1_config,
                        reflection_model_config=self.reflection_model_config,
                        max_iterations=self.max_iterations,
                        output_dir=self.output_dir,
                        enable_evaluation=True,
                        include_problem_text_description=self.include_problem_text_description,
                    )
                else:
                    from agents.langgraph_agent import LangGraphFlowchartAgent

                    agent = LangGraphFlowchartAgent(
                        data_root=self.data_root,
                        vision_model_2_config=self.vision_model_2_config,
                        language_model_1_config=self.language_model_1_config,
                        output_dir=self.output_dir,
                        enable_evaluation=True,
                        include_problem_text_description=self.include_problem_text_description,
                    )
                
                samples_file = Path(self.output_dir) / dataset_name / self.model_name / "samples.jsonl"
                results, _ = self._run_with_optional_output_capture(
                    dataset_name,
                    "evaluation",
                    lambda: agent.evaluate_samples(
                        sample_file=str(samples_file),
                        n_workers=4,
                        timeout=3.0,
                        k=[1]
                    ),
                )
                
                print(f"✓ Evaluation complete")
                print(f"  Pass@1: {results.get('pass@1', 0):.4f}")

                # Generate detailed statistics
                detailed_stats = self._generate_detailed_statistics(dataset_name)

                # Print statistics summary
                if detailed_stats:
                    self._print_statistics_summary(detailed_stats)

                # Update progress
                self._update_dataset_progress(
                    dataset_name, dataset_hash,
                    "completed",
                    f"Evaluation complete: Pass@1 = {results.get('pass@1', 0):.4f}"
                )

                return True
            except Exception as e:
                print(f"✗ Evaluation failed: {e}")
                import traceback
                traceback.print_exc()
                self._update_dataset_progress(
                    dataset_name, dataset_hash,
                    "evaluation_failed",
                    f"Evaluation failed: {e}"
                )
                return False
        
        return True

    def _generate_detailed_statistics(self, dataset_name: str) -> Dict[str, Any]:
        """
        Generate detailed statistics

        Includes:
        1. Pass rate without iteration (by difficulty)
        2. Total pass rate after 1, 2, 3 iterations
        3. Token and time statistics per iteration
        4. Average tokens and time per sample
        """
        samples_file = Path(self.output_dir) / dataset_name / self.model_name / "samples.jsonl"
        results_file = Path(self.output_dir) / dataset_name / self.model_name / "samples.jsonl_results.jsonl"

        if not samples_file.exists() or not results_file.exists():
            print(f"⚠ Statistics files not found, skipping statistics")
            return {}

        # Read samples and results
        samples = []
        with open(samples_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line.strip()))

        results = []
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line.strip()))

        # Merge samples and results
        task_data = {}
        for sample in samples:
            task_id = sample.get('task_id')
            if task_id:
                task_data[task_id] = {
                    'sample': sample,
                    'result': None
                }

        for result in results:
            task_id = result.get('task_id')
            if task_id and task_id in task_data:
                task_data[task_id]['result'] = result

        # Initialize statistics
        stats = {
            "dataset": dataset_name,
            "model": self.model_name,
            "total_samples": len(task_data),
            "use_reflection": self.use_reflection,
            "max_iterations": self.max_iterations if self.use_reflection else 0
        }

        if self.use_reflection:
            # Statistics with reflection
            stats.update(self._calculate_reflection_statistics(task_data))
        else:
            # Statistics without reflection
            stats.update(self._calculate_no_reflection_statistics(task_data))

        # Save statistics results
        stats_file = Path(self.output_dir) / dataset_name / self.model_name / "detailed_statistics.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)

        print(f"✓ Detailed statistics saved to: {stats_file}")

        return stats

    def _calculate_no_reflection_statistics(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics without reflection"""
        stats = {}

        # Count pass rate
        total = len(task_data)
        passed = sum(1 for data in task_data.values() if data['result'] and data['result'].get('passed', False))

        stats['pass_rate'] = {
            "iteration_0": {
                "passed": passed,
                "total": total,
                "rate": round(passed / total, 4) if total > 0 else 0
            }
        }

        # Count tokens
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        valid_samples = 0

        for data in task_data.values():
            sample = data['sample']
            usage = sample.get('usage', {})
            if usage and 'total' in usage:
                total_usage = usage['total']
                total_prompt_tokens += total_usage.get('prompt_tokens', 0)
                total_completion_tokens += total_usage.get('completion_tokens', 0)
                total_tokens += total_usage.get('total_tokens', 0)
                valid_samples += 1

        stats['tokens'] = {
            "total": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens
            },
            "average_per_sample": {
                "prompt_tokens": round(total_prompt_tokens / valid_samples, 2) if valid_samples > 0 else 0,
                "completion_tokens": round(total_completion_tokens / valid_samples, 2) if valid_samples > 0 else 0,
                "total_tokens": round(total_tokens / valid_samples, 2) if valid_samples > 0 else 0
            }
        }

        # Count time
        total_time = 0
        valid_time_samples = 0

        for data in task_data.values():
            sample = data['sample']
            time_info = sample.get('time_info', {})
            if time_info and 'total_time' in time_info:
                total_time += time_info['total_time']
                valid_time_samples += 1

        stats['time'] = {
            "total_time": round(total_time, 2),
            "average_per_sample": round(total_time / valid_time_samples, 2) if valid_time_samples > 0 else 0
        }

        # Statistics by difficulty (if difficulty field exists)
        difficulty_stats = {}
        for data in task_data.values():
            sample = data['sample']
            difficulty = sample.get('difficulty', 'unknown')

            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {
                    'total': 0,
                    'passed': 0
                }

            difficulty_stats[difficulty]['total'] += 1
            if data['result'] and data['result'].get('passed', False):
                difficulty_stats[difficulty]['passed'] += 1

        # Calculate pass rate for each difficulty
        for difficulty, counts in difficulty_stats.items():
            counts['rate'] = round(counts['passed'] / counts['total'], 4) if counts['total'] > 0 else 0

        if len(difficulty_stats) > 1 or 'unknown' not in difficulty_stats:
            stats['pass_rate_by_difficulty'] = difficulty_stats

        return stats

    def _calculate_reflection_statistics(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate statistics with reflection"""
        stats = {}

        # Group by iteration
        by_iteration = {
            0: [],  # Initial generation
            1: [],  # 1st reflection
            2: [],  # 2nd reflection
            3: []   # 3rd reflection
        }

        for task_id, data in task_data.items():
            sample = data['sample']
            result = data['result']
            iteration = sample.get('iteration', 0)

            if iteration in by_iteration:
                by_iteration[iteration].append({
                    'task_id': task_id,
                    'sample': sample,
                    'result': result,
                    'passed': result.get('passed', False) if result else False
                })

        # Count pass rate per iteration
        pass_rate_stats = {}
        cumulative_passed = set()  # Cumulative passed tasks

        for iteration in sorted(by_iteration.keys()):
            samples = by_iteration[iteration]
            if not samples:
                continue

            # Tasks passed in current iteration
            passed_tasks = {s['task_id'] for s in samples if s['passed']}
            cumulative_passed.update(passed_tasks)

            # Statistics by difficulty for current iteration
            difficulty_stats_iter = {}
            for s in samples:
                difficulty = s['sample'].get('difficulty', 'unknown')
                if difficulty not in difficulty_stats_iter:
                    difficulty_stats_iter[difficulty] = {
                        'total': 0,
                        'passed': 0
                    }
                difficulty_stats_iter[difficulty]['total'] += 1
                if s['passed']:
                    difficulty_stats_iter[difficulty]['passed'] += 1

            # Calculate pass rate for each difficulty
            for difficulty, counts in difficulty_stats_iter.items():
                counts['rate'] = round(counts['passed'] / counts['total'], 4) if counts['total'] > 0 else 0

            pass_rate_stats[f"iteration_{iteration}"] = {
                "samples_at_this_iteration": len(samples),
                "passed_at_this_iteration": len(passed_tasks),
                "rate_at_this_iteration": round(len(passed_tasks) / len(samples), 4) if samples else 0,
                "cumulative_passed": len(cumulative_passed),
                "cumulative_rate": round(len(cumulative_passed) / len(task_data), 4) if task_data else 0
            }

            # Only add difficulty statistics when there are multiple difficulties or not all 'unknown'
            if len(difficulty_stats_iter) > 1 or 'unknown' not in difficulty_stats_iter:
                pass_rate_stats[f"iteration_{iteration}"]["by_difficulty"] = difficulty_stats_iter

        stats['pass_rate'] = pass_rate_stats

        # Count tokens (by iteration)
        tokens_by_iteration = {}

        for iteration in sorted(by_iteration.keys()):
            samples = by_iteration[iteration]
            if not samples:
                continue

            total_prompt = 0
            total_completion = 0
            total_tokens = 0

            for s in samples:
                usage = s['sample'].get('usage', {})
                if 'by_iteration' in usage and f"iteration_{iteration}" in usage['by_iteration']:
                    iter_usage = usage['by_iteration'][f"iteration_{iteration}"]
                    total_prompt += iter_usage.get('prompt_tokens', 0)
                    total_completion += iter_usage.get('completion_tokens', 0)
                    total_tokens += iter_usage.get('total_tokens', 0)

            tokens_by_iteration[f"iteration_{iteration}"] = {
                "total": {
                    "prompt_tokens": total_prompt,
                    "completion_tokens": total_completion,
                    "total_tokens": total_tokens
                },
                "average_per_sample": {
                    "prompt_tokens": round(total_prompt / len(samples), 2) if samples else 0,
                    "completion_tokens": round(total_completion / len(samples), 2) if samples else 0,
                    "total_tokens": round(total_tokens / len(samples), 2) if samples else 0
                }
            }

        # Total tokens
        total_prompt = 0
        total_completion = 0
        total_tokens_sum = 0

        for data in task_data.values():
            usage = data['sample'].get('usage', {})
            if 'total' in usage:
                total_usage = usage['total']
                total_prompt += total_usage.get('prompt_tokens', 0)
                total_completion += total_usage.get('completion_tokens', 0)
                total_tokens_sum += total_usage.get('total_tokens', 0)

        tokens_by_iteration['total_all_iterations'] = {
            "total": {
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "total_tokens": total_tokens_sum
            },
            "average_per_sample": {
                "prompt_tokens": round(total_prompt / len(task_data), 2) if task_data else 0,
                "completion_tokens": round(total_completion / len(task_data), 2) if task_data else 0,
                "total_tokens": round(total_tokens_sum / len(task_data), 2) if task_data else 0
            }
        }

        stats['tokens'] = tokens_by_iteration

        # Count time (by iteration)
        time_by_iteration = {}

        for iteration in sorted(by_iteration.keys()):
            samples = by_iteration[iteration]
            if not samples:
                continue

            total_time = 0

            for s in samples:
                time_info = s['sample'].get('time_info', {})
                if 'iteration_times' in time_info and f"iteration_{iteration}" in time_info['iteration_times']:
                    total_time += time_info['iteration_times'][f"iteration_{iteration}"]

            time_by_iteration[f"iteration_{iteration}"] = {
                "total_time": round(total_time, 2),
                "average_per_sample": round(total_time / len(samples), 2) if samples else 0
            }

        # Total time
        total_time_all = 0
        for data in task_data.values():
            time_info = data['sample'].get('time_info', {})
            if 'total_time' in time_info:
                total_time_all += time_info['total_time']

        time_by_iteration['total_all_iterations'] = {
            "total_time": round(total_time_all, 2),
            "average_per_sample": round(total_time_all / len(task_data), 2) if task_data else 0
        }

        stats['time'] = time_by_iteration

        # Statistics by difficulty (if available)
        difficulty_stats = {}
        for data in task_data.values():
            sample = data['sample']
            difficulty = sample.get('difficulty', 'unknown')

            if difficulty not in difficulty_stats:
                difficulty_stats[difficulty] = {
                    'total': 0,
                    'passed': 0
                }

            difficulty_stats[difficulty]['total'] += 1
            if data['result'] and data['result'].get('passed', False):
                difficulty_stats[difficulty]['passed'] += 1

        for difficulty, counts in difficulty_stats.items():
            counts['rate'] = round(counts['passed'] / counts['total'], 4) if counts['total'] > 0 else 0

        if len(difficulty_stats) > 1 or 'unknown' not in difficulty_stats:
            stats['pass_rate_by_difficulty'] = difficulty_stats

        return stats

    def _print_statistics_summary(self, stats: Dict[str, Any]) -> None:
        """Print statistics summary"""
        print(f"\n{'='*70}")
        print(f"Detailed Statistics Summary")
        print(f"{'='*70}")

        if stats.get('use_reflection'):
            # Statistics with reflection
            print(f"Mode: With reflection (max iterations: {stats.get('max_iterations', 0)})")
            print(f"\nPass rate statistics:")

            pass_rate = stats.get('pass_rate', {})
            for iteration_key in sorted(pass_rate.keys()):
                iter_stats = pass_rate[iteration_key]
                print(f"  {iteration_key}:")
                print(f"    - This iteration: {iter_stats.get('passed_at_this_iteration', 0)}/{iter_stats.get('samples_at_this_iteration', 0)} "
                      f"({iter_stats.get('rate_at_this_iteration', 0):.2%})")
                print(f"    - Cumulative passed: {iter_stats.get('cumulative_passed', 0)}/{stats.get('total_samples', 0)} "
                      f"({iter_stats.get('cumulative_rate', 0):.2%})")

                # Display statistics by difficulty
                if 'by_difficulty' in iter_stats:
                    print(f"    - By difficulty:")
                    for difficulty, counts in iter_stats['by_difficulty'].items():
                        print(f"      {difficulty}: {counts['passed']}/{counts['total']} ({counts['rate']:.2%})")

            print(f"\nToken statistics:")
            tokens = stats.get('tokens', {})
            for iteration_key in sorted([k for k in tokens.keys() if k.startswith('iteration_')]):
                iter_tokens = tokens[iteration_key]
                print(f"  {iteration_key}:")
                print(f"    - Total: {iter_tokens['total']['total_tokens']:,} tokens")
                print(f"    - Average: {iter_tokens['average_per_sample']['total_tokens']:.2f} tokens/sample")

            if 'total_all_iterations' in tokens:
                total_tokens = tokens['total_all_iterations']
                print(f"  All iterations total:")
                print(f"    - Total: {total_tokens['total']['total_tokens']:,} tokens")
                print(f"    - Average: {total_tokens['average_per_sample']['total_tokens']:.2f} tokens/sample")

            print(f"\nTime statistics:")
            time_stats = stats.get('time', {})
            for iteration_key in sorted([k for k in time_stats.keys() if k.startswith('iteration_')]):
                iter_time = time_stats[iteration_key]
                print(f"  {iteration_key}:")
                print(f"    - Total: {iter_time['total_time']:.2f}s")
                print(f"    - Average: {iter_time['average_per_sample']:.2f}s/sample")

            if 'total_all_iterations' in time_stats:
                total_time = time_stats['total_all_iterations']
                print(f"  All iterations total:")
                print(f"    - Total: {total_time['total_time']:.2f}s")
                print(f"    - Average: {total_time['average_per_sample']:.2f}s/sample")

        else:
            # Statistics without reflection
            print(f"Mode: Without reflection")

            pass_rate = stats.get('pass_rate', {}).get('iteration_0', {})
            print(f"\nPass rate: {pass_rate.get('passed', 0)}/{pass_rate.get('total', 0)} ({pass_rate.get('rate', 0):.2%})")

            tokens = stats.get('tokens', {})
            print(f"\nToken statistics:")
            print(f"  Total: {tokens['total']['total_tokens']:,} tokens")
            print(f"  Average: {tokens['average_per_sample']['total_tokens']:.2f} tokens/sample")

            time_stats = stats.get('time', {})
            print(f"\nTime statistics:")
            print(f"  Total: {time_stats['total_time']:.2f}s")
            print(f"  Average: {time_stats['average_per_sample']:.2f}s/sample")

        # Statistics by difficulty
        if 'pass_rate_by_difficulty' in stats:
            print(f"\nStatistics by difficulty:")
            for difficulty, counts in stats['pass_rate_by_difficulty'].items():
                print(f"  {difficulty}: {counts['passed']}/{counts['total']} ({counts['rate']:.2%})")

        print(f"{'='*70}\n")

    def run_all(self) -> Dict[str, Any]:
        """Run evaluation for all datasets"""
        print(f"{'='*70}")
        print(f"Starting batch evaluation")
        print(f"{'='*70}")
        print(f"Model name: {self.model_name}")
        print(f"Data root directory: {self.data_root}")
        print(f"Output directory: {self.output_dir}")
        print(f"Progress file: {self.progress_file}")
        print(f"Stream output: {'enabled' if self.enable_stream_output else 'disabled'}")
        
        results = {
            "model_name": self.model_name,
            "datasets": {},
            "summary": {
                "total": len(self.datasets),
                "completed": 0,
                "failed": 0,
                "skipped": 0
            }
        }
        
        for dataset in self.datasets:
            dataset_name = dataset["name"]
            
            try:
                success = self.run_dataset(dataset)
                
                if success:
                    results["datasets"][dataset_name] = {
                        "status": "completed",
                        "message": "Evaluation complete"
                    }
                    results["summary"]["completed"] += 1
                else:
                    results["datasets"][dataset_name] = {
                        "status": "failed",
                        "message": "Evaluation failed"
                    }
                    results["summary"]["failed"] += 1
                    
            except Exception as e:
                print(f"✗ Dataset {dataset_name} processing exception: {e}")
                import traceback
                traceback.print_exc()
                
                results["datasets"][dataset_name] = {
                    "status": "failed",
                    "message": f"Processing exception: {e}"
                }
                results["summary"]["failed"] += 1
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Batch evaluation complete")
        print(f"{'='*70}")
        print(f"Total datasets: {results['summary']['total']}")
        print(f"Completed: {results['summary']['completed']}")
        print(f"Failed: {results['summary']['failed']}")
        print(f"Skipped: {results['summary']['skipped']}")
        
        # Save summary results
        summary_file = self.progress_file.parent / f"{self.model_name}_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Summary results saved to: {summary_file}")
        
        return results


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description='Batch evaluation script (LangGraph multi-agent) - Supports checkpoint resume'
    )
    parser.add_argument(
        '--vision-model-2',
        type=str,
        default=None,
        help='Vision model 2 config file path (for pseudocode generation)'
    )
    parser.add_argument(
        '--language-model-1',
        type=str,
        default=None,
        help='Language model 1 config file path (for code generation)'
    )
    parser.add_argument(
        '--single-model-configs',
        type=str,
        nargs='+',
        default=None,
        help='Run each config file as both vision_model_2 and language_model_1'
    )
    parser.add_argument(
        '--reflection-model',
        type=str,
        default=None,
        help='Reflection model config file path (optional, for reflection iteration)'
    )
    parser.add_argument(
        '--use-reflection',
        action='store_true',
        help='Enable reflection functionality (requires --reflection-model)'
    )
    parser.add_argument(
        '--max-iterations',
        type=int,
        default=3,
        help='Maximum iteration count (default 3, only effective when reflection is enabled)'
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=None,
        help='Data root directory (default: ../data)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: ../output)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        nargs='+',
        default=None,
        help='One or more datasets to run (default: run all)'
    )
    parser.add_argument(
        '--reset',
        action='store_true',
        help='Reset progress and restart evaluation'
    )
    parser.add_argument(
        '--intermediate-representation',
        type=str,
        default='ilr',
        choices=['ilr', 'text'],
        help='Agent1 intermediate representation output type: ilr or text (for ablation study)'
    )
    parser.add_argument(
        '--prompt-variant',
        type=str,
        default='default',
        choices=['default', 'self_planning'],
        help='Prompt variant: default or self_planning'
    )
    parser.add_argument(
        '--stream-output',
        action='store_true',
        help='Enable model streaming output, print generated content in real-time in terminal'
    )
    parser.add_argument(
        '--include-problem-text-description',
        type=str2bool,
        nargs='?',
        const=True,
        default=True,
        help='Whether to include the problem text description from the dataset as additional context for ILR/reflection process, default true; pass false to disable'
    )
    
    args = parser.parse_args()

    # Check reflection functionality parameters
    if args.use_reflection and not args.reflection_model:
        parser.error("Must provide --reflection-model parameter when using reflection functionality")
    if not args.single_model_configs and (not args.vision_model_2 or not args.language_model_1):
        parser.error(
            "Must provide --vision-model-2 and --language-model-1, or use --single-model-configs"
        )

    selected_datasets = args.dataset

    def create_evaluator(vision_model_2: str, language_model_1: str) -> ResumableBatchEvaluator:
        return ResumableBatchEvaluator(
            vision_model_2_config=vision_model_2,
            language_model_1_config=language_model_1,
            reflection_model_config=args.reflection_model,
            use_reflection=args.use_reflection,
            max_iterations=args.max_iterations,
            intermediate_representation_type=args.intermediate_representation,
            prompt_variant=args.prompt_variant,
            data_root=args.data_root,
            output_dir=args.output_dir,
            dataset_names=selected_datasets,
            enable_stream_output=args.stream_output,
            include_problem_text_description=args.include_problem_text_description,
        )

    def maybe_reset_progress(evaluator: ResumableBatchEvaluator) -> None:
        if args.reset and evaluator.progress_file.exists():
            evaluator.progress_file.unlink()
            print(f"Progress file reset: {evaluator.progress_file}")

    if args.single_model_configs:
        overall_failed = 0

        for config_path in args.single_model_configs:
            print(f"\n{'#' * 70}")
            print(f"Starting single-config mode: {config_path}")
            print(f"{'#' * 70}")

            evaluator = create_evaluator(config_path, config_path)
            maybe_reset_progress(evaluator)
            results = evaluator.run_all()

            if results["summary"]["failed"] > 0:
                overall_failed += results["summary"]["failed"]

        if overall_failed > 0:
            print("\n⚠ Warning: Some config evaluations failed, please check logs")
            sys.exit(1)

        print("\n✓ All config evaluations complete")
        sys.exit(0)

    evaluator = create_evaluator(args.vision_model_2, args.language_model_1)
    maybe_reset_progress(evaluator)

    results = evaluator.run_all()

    if results["summary"]["failed"] > 0:
        print("\n⚠ Warning: Some dataset evaluations failed, please check logs")
        sys.exit(1)

    print("\n✓ All dataset evaluations complete")
    sys.exit(0)


if __name__ == "__main__":
    main()
