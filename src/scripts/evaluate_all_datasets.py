#!/usr/bin/env python3
"""
Universal batch evaluation script for src
Supports automatic evaluation of three datasets (HumanEval-V, Algorithm, MATH) based on the provided config
Each dataset is evaluated and results are saved immediately after completion
"""

import sys
import os
import json
import argparse
import time
from pathlib import Path
from typing import Dict, List, Any
import hashlib
from tqdm import tqdm

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.evaluate_lcb import evaluate_lcb_samples, save_lcb_results


class UniversalBatchEvaluator:
    """Universal batch evaluator"""
    
    def __init__(self, api_config: str, data_root: str = None, output_dir: str = None,
                 prompt_variant: str = "default", enable_stream_output: bool = False,
                 text_image: bool = False, dataset_names: List[str] = None):
        """
        Initialize the evaluator

        Args:
            api_config: API config file path
            data_root: Data root directory
            output_dir: Output directory
            text_image: Whether to pass both problem description and image to the model
            dataset_names: Optional list of dataset names to run (default: all)
        """
        # Load config file
        self.api_config = self._load_config(api_config)
        self._configure_api_endpoint()

        # Stream output
        self.enable_stream_output = enable_stream_output
        if enable_stream_output:
            self.api_config["stream"] = True

        # text+image mode
        self.text_image = text_image

        # Set paths
        if data_root is None:
            data_root = str(Path(__file__).parent.parent.parent / "data")
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent.parent / "output")

        self.data_root = data_root
        self.output_dir = output_dir
        self.prompt_variant = (prompt_variant or "default").strip().lower()

        # Get model name (for output folder)
        self.model_name = self.api_config.get("model", "model").replace("/", "-")
        if self.text_image:
            self.model_name += "_text_image"
        elif self.prompt_variant == "self_planning":
            self.model_name += "_self_planning"
        elif self.prompt_variant == "zero_shot_cot":
            self.model_name += "_zero_shot_cot"

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
                "original_data_file": "LiveCodeBench_merged.jsonl",  # Merged LCB data with call_based test cases for AtCoder problems
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

    def _configure_api_endpoint(self) -> None:
        """
        Detect whether the config already provides a complete API endpoint path.

        When base_url explicitly points to `/v1/messages`, subsequent requests should
        use that URL directly without auto-appending `/chat/completions`.
        """
        base_url = str(self.api_config.get("base_url", "")).rstrip("/")
        if base_url.endswith("/v1/messages"):
            self.api_config["use_direct_messages_url"] = True
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load config file"""
        if not Path(config_path).exists():
            # Try to find in configs directory
            config_path = Path(__file__).parent.parent / "configs" / config_path
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
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
        """Compute dataset hash for detecting dataset changes"""
        data_file = Path(self.data_root) / dataset["name"] / dataset["data_file"]
        images_dir = Path(self.data_root) / dataset["name"] / dataset["images_dir"]
        
        if not data_file.exists() or not images_dir.exists():
            return None
        
        # Compute data file hash
        with open(data_file, 'rb') as f:
            data_hash = hashlib.md5(f.read()).hexdigest()
        
        # Compute image directory hash (based on file list and modification time)
        image_files = sorted(images_dir.glob("*.png"))
        image_info = []
        for img_file in image_files:
            stat = img_file.stat()
            image_info.append(f"{img_file.name}:{stat.st_size}:{stat.st_mtime}")
        image_hash = hashlib.md5("|".join(image_info).encode()).hexdigest()
        
        return f"{data_hash}:{image_hash}"
    
    def _check_dataset_completed(self, dataset_name: str, dataset_hash: str) -> bool:
        """Check if the dataset is completed"""
        progress = self._load_progress()
        
        if dataset_name not in progress["datasets"]:
            return False
        
        dataset_progress = progress["datasets"][dataset_name]
        
        # Check if hash matches
        if dataset_progress.get("hash") != dataset_hash:
            print(f"  [WARNING] Dataset {dataset_name} has changed, needs re-evaluation")
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
        """Check if samples.jsonl is completed"""
        samples_file = Path(self.output_dir) / dataset_name / self.model_name / "samples.jsonl"
        
        if not samples_file.exists():
            return False
        
        # Read samples.jsonl with error handling for encoding issues
        samples = []
        with open(samples_file, 'r', encoding='utf-8', errors='replace') as f:
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
            response_str = sample.get("response", "{}")
            if not response_str or response_str.strip() == "":
                response_str = "{}"
            try:
                response = json.loads(response_str)
            except json.JSONDecodeError:
                response = {}
            if response.get("status") == "failed":
                return False
        
        return True
    
    def _check_evaluation_completed(self, dataset_name: str) -> bool:
        """Check if evaluation is completed"""
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
        if self.text_image:
            print(f"Mode: text+image (problem description + image)")

        # Compute dataset hash
        dataset_hash = self._get_dataset_hash(dataset)
        if dataset_hash is None:
            print(f"[ERROR] Dataset {dataset_name} file not found, skipping")
            return False
        
        # Check if already completed
        if self._check_dataset_completed(dataset_name, dataset_hash):
            print(f"[OK] Dataset {dataset_name} already completed, skipping")
            return True
        
        # Get image count
        image_files = list(images_dir.glob("*.png"))
        total_images = len(image_files)
        
        if total_images == 0:
            print(f"[ERROR] Dataset {dataset_name} has no images, skipping")
            return False
        
        print(f"Total images: {total_images}")
        
        # Check if samples.jsonl is completed
        samples_completed = self._check_samples_completed(dataset_name, total_images)
        if samples_completed:
            print(f"[OK] samples.jsonl completed ({total_images} samples)")
        else:
            print(f"[INFO] Starting code generation...")
            # Generate code
            try:
                from evaluation.evaluate_baseline import (
                    read_jsonl_file,
                    write_jsonl_file,
                    extract_code,
                    extract_code_block_only,
                    run_two_stage_self_planning,
                    normalize_response_tuple,
                )
                from evaluation.evaluate_baseline import (
                    get_response,
                    construct_prompt,
                    construct_prompt_cot,
                )
                # Import prompt construction functions for text mode
                from evaluation.evaluate_text import (
                    construct_prompt_text,
                    construct_prompt_cot_text,
                )
                
                # Read problem data
                problems = read_jsonl_file(str(data_file))
                
                # Check already processed results
                output_dir = os.path.join(self.output_dir, dataset_name, self.model_name)
                os.makedirs(output_dir, exist_ok=True)
                save_path = os.path.join(output_dir, "samples.jsonl")
                
                done_data = []
                if os.path.exists(save_path):
                    done_data = read_jsonl_file(save_path)
                
                done_ids = [x["task_id"] for x in done_data]
                
                # Filter already processed problems
                if done_ids:
                    print(f"Skipping {len(done_ids)} already processed tasks")
                    problems = [x for x in problems if x["task_id"] not in done_ids]
                    print(f"Remaining {len(problems)} tasks to process")
                
                # Process each problem
                with open(save_path, "a", encoding='utf-8') as f:
                    for problem in tqdm(problems, desc=f"Generating {dataset_name}", total=len(problems), ascii=True):
                        task_id = problem["task_id"]
                        image_path = os.path.join(images_dir, task_id + ".png")
                        
                        try:
                            # Get problem description
                            problem_description = problem.get("prompt", "").strip()

                            # Construct prompt
                            if self.text_image:
                                # text+image mode: use prompt with problem description, and pass image
                                if self.prompt_variant == "zero_shot_cot":
                                    prompt = construct_prompt_cot_text(problem)
                                else:
                                    prompt = construct_prompt_text(problem)
                            elif self.prompt_variant == "zero_shot_cot":
                                # Zero-Shot-CoT mode: pass problem description
                                prompt = construct_prompt_cot(problem["starter_code"], problem_description)
                            else:
                                prompt = construct_prompt(problem["starter_code"])

                            plan_text = ""
                            plan_usage = None

                            # Record start time
                            start_time = time.time()

                            # Call API
                            try:
                                total_retry_duration = 0
                                if not self.text_image and self.prompt_variant == "self_planning":
                                    # self_planning mode: pass problem description
                                    planning_result = run_two_stage_self_planning(
                                        problem["starter_code"],
                                        image_path,
                                        self.api_config,
                                        lambda current_prompt, current_image_path: get_response(
                                            current_prompt,
                                            current_image_path,
                                            self.api_config,
                                        ),
                                        description=problem_description,  # Pass problem description
                                    )
                                    responses = [planning_result["code_response"]]
                                    usage = planning_result["usage"]
                                    model_name = planning_result["model"]
                                    total_retry_duration += planning_result["retry_duration"]
                                    plan_text = planning_result["plan"]
                                    plan_usage = planning_result["plan_usage"]
                                    code_usage = planning_result["code_usage"]
                                    plan_response = planning_result["plan_response"]
                                else:
                                    # text_image mode or default/zero_shot_cot mode: call API directly
                                    responses, usage, model_name, retry_duration = normalize_response_tuple(
                                        get_response(prompt, image_path, self.api_config),
                                        default_model_name=self.api_config.get("model", "")
                                    )
                                    total_retry_duration += retry_duration
                            except Exception as e:
                                raise e

                            # Record end time and calculate duration
                            end_time = time.time()
                            duration = end_time - start_time - total_retry_duration
                            if duration < 0:
                                duration = 0

                            # Extract code
                            if self.prompt_variant == "zero_shot_cot":
                                completion = extract_code_block_only(responses[0])
                            else:
                                completion = extract_code(responses[0])

                            # Save results
                            result = problem.copy()
                            result["response"] = responses[0]
                            result["completion"] = completion
                            result["usage"] = usage
                            if plan_text:
                                result["plan"] = plan_text
                            if plan_usage is not None:
                                result["plan_usage"] = plan_usage
                            if self.prompt_variant == "self_planning":
                                if code_usage is not None:
                                    result["code_usage"] = code_usage
                                if plan_response:
                                    result["plan_response"] = plan_response
                            result["model"] = model_name
                            result["duration"] = duration  # Record generation duration (seconds)

                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f.flush()

                            # Add request delay to prevent 429 rate limiting
                            # Default wait is 2 seconds, adjustable via request_delay in config
                            request_delay = self.api_config.get("request_delay", 2.0)
                            if request_delay > 0:
                                time.sleep(request_delay)

                        except Exception as e:
                            error_msg = str(e).encode('ascii', 'replace').decode('ascii')
                            print(f"[ERROR] Task {task_id} failed: {error_msg}")
                            if "UNRECOVERABLE_BACKEND_ERROR" in str(e):
                                raise RuntimeError(str(e))
                            # Save error result
                            result = problem.copy()
                            result["response"] = json.dumps({"status": "failed", "error": str(e)}, ensure_ascii=False)
                            result["completion"] = ""
                            result["usage"] = {}
                            result["model"] = self.model_name

                            f.write(json.dumps(result, ensure_ascii=False) + "\n")
                            f.flush()

                print(f"[OK] Code generation completed")
                
                # Update progress
                self._update_dataset_progress(
                    dataset_name, dataset_hash,
                    "samples_completed",
                    f"Code generation completed"
                )
                
            except Exception as e:
                error_msg = str(e).encode('ascii', 'replace').decode('ascii')
                print(f"[ERROR] Code generation failed: {error_msg}")
                import traceback
                traceback.print_exc()
                self._update_dataset_progress(
                    dataset_name, dataset_hash,
                    "samples_failed",
                    f"Code generation failed: {e}"
                )
                return False
        
        # Check if evaluation is completed
        evaluation_completed = self._check_evaluation_completed(dataset_name)
        if evaluation_completed:
            print(f"[OK] Evaluation completed")
            # Read pass@1 and save results file
            results_file = Path(self.output_dir) / dataset_name / self.model_name / "samples.jsonl_results.jsonl"
            if results_file.exists():
                results = []
                with open(results_file, 'r', encoding='utf-8', errors='replace') as f:
                    for line in f:
                        try:
                            results.append(json.loads(line.strip()))
                        except:
                            continue
                total = len(results)
                passed = sum(1 for r in results if r.get("passed", False))
                pass_at_1 = passed / total if total > 0 else 0
                self._save_results_file(dataset_name, pass_at_1)
        else:
            print(f"[INFO] Starting evaluation...")
            # Evaluate code
            try:
                samples_file = Path(self.output_dir) / dataset_name / self.model_name / "samples.jsonl"
                
                if dataset_name == "LiveCodeBench":
                    # Use LCB official evaluation
                    print("Running LCB official evaluation...")
                    # Use original LCB data file which contains test cases
                    original_data_file = Path(self.data_root) / dataset_name / dataset.get("original_data_file", dataset["data_file"])
                    pass_at_1 = self._evaluate_lcb(str(samples_file), str(original_data_file))
                    print(f"[OK] Evaluation completed")
                    print(f"  Pass@1: {pass_at_1:.4f}")
                else:
                    from human_eval.evaluation import evaluate_functional_correctness
                    
                    # Run evaluation
                    print("Running evaluation...")
                    results = evaluate_functional_correctness(
                        sample_file=str(samples_file),
                        k=[1],
                        n_workers=4,
                        timeout=3.0,
                        problem_file=str(data_file)
                    )
                    
                    print(f"[OK] Evaluation completed")
                    print(f"  Pass@1: {results.get('pass@1', 0):.4f}")
                    pass_at_1 = results.get('pass@1', 0)
                
                # Calculate token statistics
                self._calculate_token_statistics(dataset_name, str(samples_file))
                
                # Save results file
                self._save_results_file(dataset_name, pass_at_1)
                
                # Update progress
                self._update_dataset_progress(
                    dataset_name, dataset_hash,
                    "completed",
                    f"Evaluation completed: Pass@1 = {pass_at_1:.4f}"
                )
                
            except Exception as e:
                error_msg = str(e).encode('ascii', 'replace').decode('ascii')
                print(f"[ERROR] Evaluation failed: {error_msg}")
                import traceback
                traceback.print_exc()
                self._update_dataset_progress(
                    dataset_name, dataset_hash,
                    "evaluation_failed",
                    f"Evaluation failed: {e}"
                )
                return False
        
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
    
    def _calculate_token_statistics(self, dataset_name: str, samples_file: str) -> None:
        """Calculate token statistics"""
        # Read sample data
        samples = []
        with open(samples_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    samples.append(json.loads(line.strip()))
                except:
                    continue
        
        # Read problem data for difficulty information
        data_file = None
        for ds in self.datasets:
            if ds["name"] == dataset_name:
                data_file = Path(self.data_root) / dataset_name / ds["data_file"]
                break
        if data_file is None:
            data_file = Path(self.data_root) / dataset_name / f"{dataset_name}.jsonl"
        
        problems = {}
        try:
            from human_eval.data import read_problems
            problems = read_problems(str(data_file))
        except Exception as e:
            print(f"Failed to read problem data: {e}")
        
        # Count tokens and time
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        valid_token_count = 0

        total_duration = 0
        valid_duration_count = 0

        # Token statistics by difficulty
        easy_tokens = 0
        easy_count = 0
        medium_tokens = 0
        medium_count = 0
        hard_tokens = 0
        hard_count = 0
        
        for sample in samples:
            task_id = sample.get("task_id", "")
            usage = sample.get("usage", {})
            duration = sample.get("duration", 0)

            if duration > 0:
                total_duration += duration
                valid_duration_count += 1

            if isinstance(usage, dict) and usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens_usage = usage.get("total_tokens", 0)
                
                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += total_tokens_usage
                valid_token_count += 1
                
                # Classify by difficulty
                difficulty = "Unknown"
                if task_id in problems:
                    meta = problems[task_id].get("meta", {})
                    difficulty = meta.get("difficulty", problems[task_id].get("difficulty", "Unknown"))
                
                difficulty = difficulty.capitalize() if difficulty and difficulty.lower() in ["easy", "medium", "hard"] else "Unknown"
                
                if difficulty == "Easy":
                    easy_tokens += total_tokens_usage
                    easy_count += 1
                elif difficulty == "Medium":
                    medium_tokens += total_tokens_usage
                    medium_count += 1
                elif difficulty == "Hard":
                    hard_tokens += total_tokens_usage
                    hard_count += 1
        
        # Calculate averages
        avg_prompt_tokens = total_prompt_tokens / valid_token_count if valid_token_count > 0 else 0
        avg_completion_tokens = total_completion_tokens / valid_token_count if valid_token_count > 0 else 0
        avg_total_tokens = total_tokens / valid_token_count if valid_token_count > 0 else 0
        avg_duration = total_duration / valid_duration_count if valid_duration_count > 0 else 0

        easy_avg = easy_tokens / easy_count if easy_count > 0 else 0
        medium_avg = medium_tokens / medium_count if medium_count > 0 else 0
        hard_avg = hard_tokens / hard_count if hard_count > 0 else 0

        # Save token statistics
        token_stats = {
            "total_tasks": valid_token_count,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "avg_total_tokens": avg_total_tokens,
            "avg_duration": avg_duration,  # Add average duration
            "by_difficulty": {
                "easy": {
                    "count": easy_count,
                    "total_tokens": easy_tokens,
                    "avg_tokens": easy_avg
                },
                "medium": {
                    "count": medium_count,
                    "total_tokens": medium_tokens,
                    "avg_tokens": medium_avg
                },
                "hard": {
                    "count": hard_count,
                    "total_tokens": hard_tokens,
                    "avg_tokens": hard_avg
                }
            }
        }
        
        token_stats_path = Path(self.output_dir) / dataset_name / self.model_name / "token_stats.json"
        with open(token_stats_path, 'w', encoding='utf-8') as f:
            json.dump(token_stats, f, indent=4, ensure_ascii=False)
        
        print(f"[OK] Token statistics saved to: {token_stats_path}")
    
    def _save_results_file(self, dataset_name: str, pass_at_1: float) -> None:
        """Save results file, including pass@1 and pass rates by difficulty"""
        results_file = Path(self.output_dir) / dataset_name / self.model_name / "samples.jsonl_results.jsonl"
        results_txt = Path(self.output_dir) / dataset_name / self.model_name / "results.txt"
        
        # Read evaluation results
        results = []
        with open(results_file, 'r', encoding='utf-8', errors='replace') as f:
            for line in f:
                try:
                    results.append(json.loads(line.strip()))
                except:
                    continue
        
        # Use Score function from score.py to generate detailed results
        try:
            from evaluation.score_baseline import Score
            score_result = Score(results)
        except:
            score_result = ""
        
        # Generate results file content
        output_lines = []
        
        # Add pass@1
        output_lines.append(f"{{'pass@1': {pass_at_1}}}")

        # Try to read and add average duration
        token_stats_path = Path(self.output_dir) / dataset_name / self.model_name / "token_stats.json"
        if token_stats_path.exists():
            try:
                with open(token_stats_path, 'r', encoding='utf-8') as f:
                    stats = json.load(f)
                    avg_duration = stats.get("avg_duration", 0)
                    if avg_duration > 0:
                        output_lines.append(f"Average Duration: {avg_duration:.4f}s")
            except:
                pass

        output_lines.append("")

        # If difficulty info is available, add pass rates by difficulty
        if score_result:
            output_lines.append(score_result.strip())
        
        # Write results file
        with open(results_txt, 'w', encoding='utf-8') as f:
            f.write("\n".join(output_lines))
        
        print(f"[OK] Results saved to: {results_txt}")
    
    def run_all(self) -> Dict[str, Any]:
        """Run evaluation for all datasets"""
        print(f"{'='*70}")
        print(f"Starting batch evaluation")
        print(f"{'='*70}")
        print(f"Model name: {self.model_name}")
        print(f"Data root: {self.data_root}")
        print(f"Output dir: {self.output_dir}")
        print(f"Progress file: {self.progress_file}")
        print(f"Stream output: {'enabled' if self.enable_stream_output else 'disabled'}")
        if self.text_image:
            print(f"Mode: text+image (problem description + image)")
        
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
                        "message": "Evaluation completed"
                    }
                    results["summary"]["completed"] += 1
                else:
                    results["datasets"][dataset_name] = {
                        "status": "failed",
                        "message": "Evaluation failed"
                    }
                    results["summary"]["failed"] += 1
                    
            except Exception as e:
                error_msg = str(e).encode('ascii', 'replace').decode('ascii')
                print(f"[ERROR] Dataset {dataset_name} processing exception: {error_msg}")
                import traceback
                traceback.print_exc()
                
                results["datasets"][dataset_name] = {
                    "status": "failed",
                    "message": f"Processing exception: {e}"
                }
                results["summary"]["failed"] += 1
        
        # Print summary
        print(f"\n{'='*70}")
        print(f"Batch evaluation completed")
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
        description='Universal batch evaluation script for src'
    )
    parser.add_argument(
        '--api-config',
        type=str,
        required=True,
        help='API config file path'
    )
    parser.add_argument(
        '--prompt-variant',
        type=str,
        default='default',
        choices=['default', 'self_planning', 'zero_shot_cot'],
        help='Prompt variant: default / self_planning / zero_shot_cot (Zero-Shot-CoT)'
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
        '--reset',
        action='store_true',
        help='Reset progress and restart evaluation'
    )
    parser.add_argument(
        '--stream-output',
        action='store_true',
        help='Enable model stream output, print generation content in terminal in real-time'
    )
    parser.add_argument(
        '--text-image',
        action='store_true',
        help='Pass both problem description and image to the model (text+image mode)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        nargs='+',
        default=None,
        help='One or more datasets to run (default: run all)'
    )

    args = parser.parse_args()

    # Create evaluator
    evaluator = UniversalBatchEvaluator(
        api_config=args.api_config,
        data_root=args.data_root,
        output_dir=args.output_dir,
        prompt_variant=args.prompt_variant,
        enable_stream_output=args.stream_output,
        text_image=args.text_image,
        dataset_names=args.dataset,
    )
    
    # If reset is specified, delete progress file
    if args.reset:
        if evaluator.progress_file.exists():
            evaluator.progress_file.unlink()
            print(f"Progress file reset: {evaluator.progress_file}")
    
    # Run evaluation
    results = evaluator.run_all()
    
    # Check for failed datasets
    if results["summary"]["failed"] > 0:
        print("\n[WARNING] Some datasets failed evaluation, please check the logs")
        sys.exit(1)
    else:
        print("\n[OK] All dataset evaluations completed")
        sys.exit(0)


if __name__ == "__main__":
    main()
