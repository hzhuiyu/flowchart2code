"""
Multi-agent system based on LangGraph
Used for processing flowchart analysis, problem extraction, and ILR generation workflow
"""

import json
import time
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from pathlib import Path
import sys

from tools.problem_extractor import ProblemExtractor
from agents.agent1_ilr_generator_with_tools import ILRGeneratorWithTools
from agents.agent2_code_generator import CodeGenerator
from utils.vision_api_client import VisionAPIClient
from utils.intermediate_representation import (
    get_intermediate_representation_display_name,
    get_intermediate_representation_file_name,
    get_intermediate_representation_subdir,
    get_intermediate_representation_suffix,
    normalize_intermediate_representation_type,
)

from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    """Agent state definition"""
    image_path: str
    dataset: str
    session_id: str
    problem_info: Optional[Dict[str, Any]]
    ocr_data: Optional[Dict[str, Any]]  # Added: OCR extracted data
    ilr: Optional[str]
    intermediate_representation_type: Optional[str]
    generated_code: Optional[str]
    error: Optional[str]
    current_step: str
    messages: Annotated[List, "Message history"]
    token_usage: Optional[Dict[str, Any]]
    start_time: Optional[float]  # Task start time


class LangGraphFlowchartAgent:
    """Flowchart processing agent based on LangGraph"""

    def __init__(self, data_root: str = None,
                 vision_model_2_config: Dict[str, Any] = None,
                 language_model_1_config: Dict[str, Any] = None,
                 output_dir: str = None,
                 enable_evaluation: bool = True,
                 intermediate_representation_type: str = "ilr",
                 prompt_variant: str = "default",
                 include_problem_text_description: bool = True):
        """Initialize the agent"""
        if data_root is None:
            data_root = str(Path(__file__).parent.parent.parent / "data")
        if output_dir is None:
            output_dir = str(Path(__file__).parent.parent.parent / "output")

        self.data_root = data_root
        self.output_dir = output_dir
        self.enable_evaluation = enable_evaluation
        self.intermediate_representation_type = normalize_intermediate_representation_type(
            intermediate_representation_type
        )
        self.prompt_variant = (prompt_variant or "default").strip().lower()
        self.include_problem_text_description = include_problem_text_description
        self.problem_extractor = ProblemExtractor(data_root)

        # Initialize API clients and components
        self.vision_api_client_ilr = VisionAPIClient(config_dict=vision_model_2_config)
        self.ilr_generator = ILRGeneratorWithTools(
            vision_model_api=self.vision_api_client_ilr,
            output_dir=self.output_dir,
            intermediate_representation_type=self.intermediate_representation_type,
            prompt_variant=self.prompt_variant,
            include_problem_text_description=self.include_problem_text_description,
        )
        self.code_generator = CodeGenerator(
            config_dict=language_model_1_config,
            output_dir=self.output_dir,
            prompt_variant=self.prompt_variant,
            include_problem_text_description=self.include_problem_text_description,
        )

        # Save model configurations
        self.vision_model_2_config = vision_model_2_config
        self.language_config = language_model_1_config

        self.graph = self._build_graph()

    def _build_graph(self):
        """Build LangGraph workflow graph"""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("start", self._start_node)
        workflow.add_node("extract_problem", self._extract_problem_node)
        workflow.add_node("generate_ilr", self._generate_ilr_node)
        workflow.add_node("generate_code", self._generate_code_node)
        workflow.add_node("handle_error", self._handle_error_node)

        # Set entry point and edges
        workflow.set_entry_point("start")
        workflow.add_conditional_edges("start", self._check_error, {"next": "extract_problem", "error": "handle_error"})
        workflow.add_conditional_edges("extract_problem", self._check_error, {"next": "generate_ilr", "error": "handle_error"})
        workflow.add_conditional_edges("generate_ilr", self._check_error, {"next": "generate_code", "error": "handle_error"})
        workflow.add_conditional_edges("generate_code", self._check_error, {"next": END, "error": "handle_error"})
        workflow.add_edge("handle_error", END)

        return workflow.compile()

    def _check_error(self, state: AgentState) -> str:
        """Check if there is an error"""
        return "error" if state.get("error") else "next"

    def _start_node(self, state: AgentState) -> AgentState:
        """Start node"""
        state["current_step"] = "start"
        state["messages"] = state.get("messages", [])

        # Initialize time recording
        if "start_time" not in state:
            state["start_time"] = time.time()

        return state

    def _extract_problem_node(self, state: AgentState) -> AgentState:
        """Problem extraction node"""
        try:
            state["current_step"] = "extract_problem"
            image_name = Path(state["image_path"]).stem
            dataset_name = state.get("dataset") or "Algorithm"
            # Restrict extraction to the requested dataset to avoid duplicate-name mismatches.
            problem_info = self.problem_extractor._find_problem_in_dataset(
                image_name, dataset_name
            )
            
            if not problem_info:
                state["error"] = f"No matching problem found: {image_name}"
            else:
                state["problem_info"] = problem_info
            return state
        except Exception as e:
            state["error"] = f"Problem extraction failed: {str(e)}"
            return state

    def _generate_ilr_node(self, state: AgentState) -> AgentState:
        """Intermediate representation generation node (with retry mechanism, always calls OCR first)."""
        state["current_step"] = "generate_ilr"
        max_retries = 2
        last_error = None
        display_name = get_intermediate_representation_display_name(
            self.intermediate_representation_type
        )

        for attempt in range(max_retries):
            try:
                # Call ILR generator (will automatically call OCR first)
                ilr, ilr_usage, ocr_data = self.ilr_generator.generate_ilr(
                    state["image_path"],
                    starter_code=state["problem_info"].get("starter_code", ""),
                    task_id=Path(state["image_path"]).stem,
                    dataset=state["dataset"],
                    problem_text_description=state["problem_info"].get("text_description", ""),
                )

                # Check if the generated ILR is valid
                if ilr and ilr.strip() and not ilr.strip().startswith("//"):
                    state["ilr"] = ilr
                    state["intermediate_representation_type"] = self.intermediate_representation_type
                    state["ocr_data"] = ocr_data  # Save OCR data to session

                    # Save token usage info
                    if not state.get("token_usage"):
                        state["token_usage"] = {}

                    # Save token usage for OCR and ILR generation
                    if 'ocr_extraction' in ilr_usage:
                        state["token_usage"]["ocr_extraction"] = ilr_usage['ocr_extraction']
                    if 'ilr_generation' in ilr_usage:
                        state["token_usage"]["ilr_generation"] = ilr_usage['ilr_generation']

                    return state
                else:
                    last_error = f"Generated {display_name} is invalid: {ilr[:100] if ilr else 'empty'}"
                    if attempt < max_retries - 1:
                        continue

            except Exception as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    continue

        # All attempts failed
        state["error"] = f"{display_name} generation failed (retried {max_retries} times): {last_error}"
        return state

    def _generate_code_node(self, state: AgentState) -> AgentState:
        """Code generation node"""
        try:
            state["current_step"] = "generate_code"
            generated_code, code_usage = self.code_generator.generate_code(
                state["ilr"],
                state["problem_info"],
                representation_type=self.intermediate_representation_type,
            )
            state["generated_code"] = generated_code
            # Save token usage info (preserving previous token_usage info)
            # Ensure token_usage exists and preserve previous info
            if "token_usage" not in state:
                state["token_usage"] = {}
            state["token_usage"]["code_generation"] = code_usage
            return state
        except Exception as e:
            state["error"] = f"Code generation failed: {str(e)}"
            return state

    def _handle_error_node(self, state: AgentState) -> AgentState:
        """Error handling node"""
        state["current_step"] = "error"
        return state

    def get_output_model_name(self) -> str:
        """Get output folder name"""
        def clean_name(name: str) -> str:
            import re
            name = re.sub(r'[^a-zA-Z0-9-]', '-', name)
            return re.sub(r'-+', '-', name).strip('-')

        vision_2 = clean_name(self.vision_model_2_config.get("model", "vision"))
        language = clean_name(self.language_config.get("model", "language"))
        suffix = get_intermediate_representation_suffix(self.intermediate_representation_type)
        name = f"{vision_2}-{language}{suffix}"
        if self.prompt_variant == "self_planning":
            name += "_self_planning"
        return name

    def _save_intermediate_results(self, task_id: str, dataset: str,
                                   ilr: str = None) -> None:
        """Save intermediate results"""
        import os
        model_name = self.get_output_model_name()
        output_dir = os.path.join(self.output_dir, dataset, model_name)
        subdir = get_intermediate_representation_subdir(self.intermediate_representation_type)
        if subdir:
            output_dir = os.path.join(output_dir, subdir)
        os.makedirs(output_dir, exist_ok=True)

        # Save ILR
        if ilr:
            ilr_file = os.path.join(
                output_dir,
                get_intermediate_representation_file_name(self.intermediate_representation_type)
            )
            with open(ilr_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps({task_id: ilr}, ensure_ascii=False) + '\n')

    def process_flowchart(self, image_path: str, dataset: str = None) -> Dict[str, Any]:
        """Main method for processing flowcharts"""
        return self._process_with_langgraph(image_path, dataset)

    def _process_with_langgraph(self, image_path: str, dataset: str = None) -> Dict[str, Any]:
        """Process using LangGraph"""
        try:
            initial_state = {
                "image_path": image_path,
                "dataset": dataset or "Algorithm",
                "session_id": f"session_{Path(image_path).stem}_{hash(image_path)}",
                "problem_info": None,
                "ocr_data": None,  # Added: OCR data
                "ilr": None,
                "intermediate_representation_type": self.intermediate_representation_type,
                "generated_code": None,
                "error": None,
                "current_step": "start",
                "messages": []
            }

            result = self.graph.invoke(initial_state)
            task_id = Path(image_path).stem
            generated_code = result.get("generated_code", "")
            problem = self._read_problem_from_dataset(task_id, dataset or "Algorithm")

            response = {
                "status": "failed" if result.get("error") else "completed",
                "problem_info": result.get("problem_info"),
                "ocr_data": result.get("ocr_data"),  # Added: includes OCR data
                "ilr": result.get("ilr"),
                "intermediate_representation": result.get("ilr"),
                "intermediate_representation_type": self.intermediate_representation_type,
                "generated_code": generated_code,
                "error": result.get("error"),
                "messages": result.get("messages", [])
            }

            model_name = self.get_output_model_name()

            if not result.get("error"):
                self._save_intermediate_results(
                    task_id=task_id,
                    dataset=dataset or "Algorithm",
                    ilr=result.get("ilr")
                )

            # Record token usage info (merge into flat structure for evaluation code compatibility)
            nested_usage = result.get("token_usage", {})
            total_usage = self._merge_token_usage(nested_usage)

            # Calculate total duration
            time_info = self._calculate_time_info(result)

            if problem:
                result_dict = problem.copy()
                result_dict["response"] = json.dumps(response, ensure_ascii=False)
                result_dict["completion"] = generated_code
                result_dict["usage"] = total_usage
                result_dict["time_info"] = time_info
                result_dict["model"] = model_name
                result_dict["ocr_data"] = result.get("ocr_data")  # Added: directly add OCR data field
                result_dict["intermediate_representation_type"] = self.intermediate_representation_type
                return result_dict
            else:
                return {
                    "task_id": task_id,
                    "response": json.dumps(response, ensure_ascii=False),
                    "completion": generated_code,
                    "usage": total_usage,
                    "time_info": time_info,
                    "model": model_name,
                    "ocr_data": result.get("ocr_data"),  # Added: directly add OCR data field
                    "intermediate_representation_type": self.intermediate_representation_type,
                }
        except Exception as e:
            task_id = Path(image_path).stem
            error_response = {
                "status": "failed",
                "error": f"LangGraph execution failed: {str(e)}",
                "generated_code": ""
            }
            problem = self._read_problem_from_dataset(task_id, dataset or "Algorithm")
            if problem:
                result_dict = problem.copy()
                result_dict["response"] = json.dumps(error_response, ensure_ascii=False)
                result_dict["completion"] = ""
                result_dict["usage"] = {}
                result_dict["model"] = self.get_output_model_name()
                return result_dict
            else:
                return {
                    "task_id": task_id,
                    "response": json.dumps(error_response, ensure_ascii=False),
                    "completion": "",
                    "usage": {},
                    "model": self.get_output_model_name()
                }

    def _read_problem_from_dataset(self, task_id: str, dataset: str) -> Optional[Dict[str, Any]]:
        """Read problem from dataset"""
        if dataset == "LiveCodeBench":
            # Merge flowchart records with official LCB metadata so samples include
            # func_name, lcb_mode, and public_test_cases.
            return self.problem_extractor.extract_problem_by_image_name(task_id)

        dataset_path = Path(self.data_root) / dataset
        if dataset == "HumanEval-V":
            jsonl_path = dataset_path / "HumanEval.jsonl"
        else:
            jsonl_path = dataset_path / f"{dataset}.jsonl"

        if not jsonl_path.exists():
            jsonl_path = dataset_path / f"{dataset}.jsonl"

        if not jsonl_path.exists():
            return None

        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    item = json.loads(line.strip())
                    if item.get('task_id') == task_id:
                        return item
        except Exception as e:
            print(f"Error reading dataset file: {e}")

        return None

    def _calculate_time_info(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate time information

        Return format:
        {
            "total_time": xxx  # Net duration from start to completion (seconds, retry wait deducted)
        }
        """
        time_info = {}
        token_usage = result.get("token_usage", {})
        total_wait_time = 0.0

        # Aggregate retry wait time across all stages (VisionAPIClient writes total_wait_time in usage)
        if isinstance(token_usage, dict):
            for usage in token_usage.values():
                if isinstance(usage, dict):
                    total_wait_time += float(usage.get("total_wait_time", 0) or 0)

        # Calculate total duration (deduct retry wait time)
        start_time = result.get("start_time")
        if start_time:
            total_time = time.time() - start_time - total_wait_time
            total_time = max(0.0, total_time)
            time_info["total_time"] = round(total_time, 2)
        else:
            time_info["total_time"] = 0

        time_info["total_wait_time"] = round(total_wait_time, 2)
        return time_info

    def _merge_token_usage(self, token_usage: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge nested token usage info, preserving both detailed and summary info

        Args:
            token_usage: Nested token usage info, e.g.:
                {
                    "ilr_generation": {"prompt_tokens": 100, "completion_tokens": 200, ...},
                    "code_generation": {"prompt_tokens": 150, "completion_tokens": 300, ...}
                }

        Returns:
            Structure with both detailed and summary info, e.g.:
                {
                    "ilr_generation": {"prompt_tokens": 100, "completion_tokens": 200, ...},
                    "code_generation": {"prompt_tokens": 150, "completion_tokens": 300, ...},
                    "total": {
                        "prompt_tokens": 250,
                        "completion_tokens": 500,
                        "total_tokens": 750,
                        "max_completion_tokens": 300
                    }
                }
        """
        merged = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "api_latency": 0,  # Added: API call pure latency
            "total_wait_time": 0  # Added: total retry wait duration
        }

        # If already a flat structure, wrap into new format
        if not any(isinstance(v, dict) for v in token_usage.values()):
            return {
                "total": token_usage
            }

        # Merge nested token info while preserving original details
        result = {}

        for stage, usage in token_usage.items():
            if isinstance(usage, dict):
                # Preserve original detailed info
                result[stage] = usage
                # Accumulate into total statistics
                merged["prompt_tokens"] += usage.get("prompt_tokens", 0)
                merged["completion_tokens"] += usage.get("completion_tokens", 0)
                merged["total_tokens"] += usage.get("total_tokens", 0)
                merged["api_latency"] += usage.get("api_latency", 0)  # Accumulate latency
                merged["total_wait_time"] += usage.get("total_wait_time", 0)

        # Add summary info
        result["total"] = merged

        return result

    def evaluate_samples(self, sample_file: str, n_workers: int = 4, 
                        timeout: float = 3.0, k: list = [1]) -> Dict[str, Any]:
        """
        Evaluate samples using the src evaluation method

        Args:
            sample_file: Sample file path
            n_workers: Number of concurrent worker threads
            timeout: Timeout duration (seconds)
            k: List of k values for pass@k

        Returns:
            Evaluation result dictionary
        """
        try:
            sample_path = Path(sample_file)
            if "LiveCodeBench" in sample_path.parts:
                # LCB contains both call-based and standard-input tasks; use its mixed I/O evaluator.
                from evaluation.evaluate_lcb import evaluate_lcb_samples, save_lcb_results

                problem_file = Path(self.data_root) / "LiveCodeBench" / "LiveCodeBench_merged.jsonl"
                pass_at_1, lcb_results = evaluate_lcb_samples(
                    str(sample_path), str(problem_file), timeout=max(timeout, 6.0)
                )
                save_lcb_results(sample_path.parent, pass_at_1, lcb_results)
                return {"pass@1": pass_at_1}

            from human_eval.evaluation import evaluate_functional_correctness
            from human_eval.data import read_problems

            # Determine problem file path
            # Infer dataset name from sample_file path
            dataset_name = None

            # Find dataset name
            for possible_dataset in ["HumanEval-V", "Algorithm", "MATH"]:
                if possible_dataset in sample_path.parts:
                    dataset_name = possible_dataset
                    break

            if dataset_name is None:
                # Default to HumanEval-V
                dataset_name = "HumanEval-V"

            # Build problem file path
            if dataset_name == "HumanEval-V":
                problem_file = str(Path(self.data_root) / dataset_name / "HumanEval.jsonl")
            else:
                problem_file = str(Path(self.data_root) / dataset_name / f"{dataset_name}.jsonl")

            # Check if problem file exists
            if not Path(problem_file).exists():
                print(f"Warning: Problem file does not exist: {problem_file}")
                return {"error": f"Problem file does not exist: {problem_file}"}

            # Read problem data
            problems = read_problems(problem_file)

            # Read sample data
            import json
            samples = []
            with open(sample_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        samples.append(json.loads(line.strip()))
                    except:
                        continue

            # Check if all problems have samples
            task_ids_in_samples = set(s.get('task_id') for s in samples)
            task_ids_in_problems = set(problems.keys())

            missing_tasks = task_ids_in_problems - task_ids_in_samples
            if missing_tasks:
                print(f"Warning: The following problems have no samples: {missing_tasks}")
                print(f"Will only evaluate problems that have samples")

            # Use the src evaluation method
            print("Starting evaluation (using src evaluation method)...")
            results = evaluate_functional_correctness(
                sample_file=sample_file,
                k=k,
                n_workers=n_workers,
                timeout=timeout,
                problem_file=problem_file
            )

            # Save evaluation results to results.txt
            self._save_evaluation_results(sample_file, problem_file, results, k)

            return results

        except Exception as e:
            print(f"Evaluation failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}

    def _save_evaluation_results(self, sample_file: str, problem_file: str, 
                                 pass_k_results: Dict[str, float], k: list) -> None:
        """
        Save evaluation results to results.txt file

        Args:
            sample_file: Sample file path
            problem_file: Problem file path
            pass_k_results: pass@k evaluation results
            k: List of k values for pass@k
        """
        import json
        from pathlib import Path

        # Read evaluation result file
        results_file = Path(sample_file + "_results.jsonl")
        if not results_file.exists():
            print(f"Warning: Evaluation result file does not exist: {results_file}")
            return

        # Read all evaluation results
        eval_results = []
        with open(results_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    eval_results.append(json.loads(line.strip()))
                except:
                    continue

        if not eval_results:
            print("Warning: No evaluation results found")
            return

        # Read problem data to get difficulty info
        problems = {}
        try:
            from human_eval.data import read_problems
            problems = read_problems(problem_file)
        except Exception as e:
            print(f"Failed to read problem data: {e}")

        # Calculate overall pass rate
        total = len(eval_results)
        passed = sum(1 for r in eval_results if r.get("passed", False))
        pass_rate = passed / total * 100 if total > 0 else 0

        # Statistics by difficulty category
        easy_passed = 0
        easy_total = 0
        medium_passed = 0
        medium_total = 0
        hard_passed = 0
        hard_total = 0

        # Token statistics
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        valid_token_count = 0

        # max_completion_tokens related statistics
        max_completion_tokens_list = []  # Record all max_completion_tokens values

        # Token statistics by difficulty category
        easy_prompt_tokens = 0
        easy_completion_tokens = 0
        easy_tokens = 0
        easy_token_count = 0

        medium_prompt_tokens = 0
        medium_completion_tokens = 0
        medium_tokens = 0
        medium_token_count = 0

        hard_prompt_tokens = 0
        hard_completion_tokens = 0
        hard_tokens = 0
        hard_token_count = 0

        # Token statistics by stage (ILR generation vs code generation)
        ilr_prompt_tokens = 0
        ilr_completion_tokens = 0
        ilr_total_tokens = 0
        ilr_count = 0

        code_prompt_tokens = 0
        code_completion_tokens = 0
        code_total_tokens = 0
        code_count = 0
        medium_completion_tokens = 0
        medium_tokens = 0
        medium_token_count = 0

        hard_prompt_tokens = 0
        hard_completion_tokens = 0
        hard_tokens = 0
        hard_token_count = 0

        for result in eval_results:
            task_id = result.get("task_id", "")
            # Get difficulty info from problem data
            difficulty = "Unknown"
            if task_id in problems:
                meta = problems[task_id].get("meta", {})
                difficulty = meta.get("difficulty", problems[task_id].get("difficulty", "Unknown"))
            else:
                # Try to get difficulty from result
                difficulty = result.get("difficulty", "Unknown")

            difficulty = difficulty.capitalize() if difficulty and difficulty.lower() in ["easy", "medium", "hard"] else "Unknown"

            # Calculate pass rate
            if difficulty == "Easy":
                easy_total += 1
                if result.get("passed", False):
                    easy_passed += 1
            elif difficulty == "Medium":
                medium_total += 1
                if result.get("passed", False):
                    medium_passed += 1
            elif difficulty == "Hard":
                hard_total += 1
                if result.get("passed", False):
                    hard_passed += 1

            # Calculate token usage (support nested and flat structures)
            usage = result.get("usage", {})
            if isinstance(usage, dict) and usage:
                # Prefer "total" field (new format), otherwise use flat structure (old format)
                if "total" in usage:
                    token_data = usage["total"]
                    # Calculate detailed token info per stage
                    if "ilr_generation" in usage:
                        ilr_usage = usage["ilr_generation"]
                        ilr_prompt_tokens += ilr_usage.get("prompt_tokens", 0)
                        ilr_completion_tokens += ilr_usage.get("completion_tokens", 0)
                        ilr_total_tokens += ilr_usage.get("total_tokens", 0)
                        ilr_count += 1

                    if "code_generation" in usage:
                        cg_usage = usage["code_generation"]
                        code_prompt_tokens += cg_usage.get("prompt_tokens", 0)
                        code_completion_tokens += cg_usage.get("completion_tokens", 0)
                        code_total_tokens += cg_usage.get("total_tokens", 0)
                        code_count += 1
                else:
                    token_data = usage
                
                prompt_tokens = token_data.get("prompt_tokens", 0)
                completion_tokens = token_data.get("completion_tokens", 0)
                total_tokens_usage = token_data.get("total_tokens", 0)
                max_completion_tokens = token_data.get("max_completion_tokens", 0)

                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += total_tokens_usage
                valid_token_count += 1

                # Calculate max_completion_tokens
                if max_completion_tokens > 0:
                    max_completion_tokens_list.append(max_completion_tokens)

                # Token statistics by difficulty category
                if difficulty == "Easy":
                    easy_prompt_tokens += prompt_tokens
                    easy_completion_tokens += completion_tokens
                    easy_tokens += total_tokens_usage
                    easy_token_count += 1
                elif difficulty == "Medium":
                    medium_prompt_tokens += prompt_tokens
                    medium_completion_tokens += completion_tokens
                    medium_tokens += total_tokens_usage
                    medium_token_count += 1
                elif difficulty == "Hard":
                    hard_prompt_tokens += prompt_tokens
                    hard_completion_tokens += completion_tokens
                    hard_tokens += total_tokens_usage
                    hard_token_count += 1

        easy_rate = easy_passed / easy_total * 100 if easy_total > 0 else 0
        medium_rate = medium_passed / medium_total * 100 if medium_total > 0 else 0
        hard_rate = hard_passed / hard_total * 100 if hard_total > 0 else 0

        # Calculate average token usage
        avg_prompt_tokens = total_prompt_tokens / valid_token_count if valid_token_count > 0 else 0
        avg_completion_tokens = total_completion_tokens / valid_token_count if valid_token_count > 0 else 0
        avg_total_tokens = total_tokens / valid_token_count if valid_token_count > 0 else 0

        easy_avg_tokens = easy_tokens / easy_token_count if easy_token_count > 0 else 0
        medium_avg_tokens = medium_tokens / medium_token_count if medium_token_count > 0 else 0
        hard_avg_tokens = hard_tokens / hard_token_count if hard_token_count > 0 else 0

        # Build results.txt content
        result_lines = []
        result_lines.append("="*60)
        result_lines.append("Evaluation Results Summary")
        result_lines.append("="*60)
        result_lines.append(f"Total samples: {total}")
        result_lines.append(f"Passed: {passed}")
        result_lines.append(f"Failed: {total - passed}")
        result_lines.append(f"Pass rate: {pass_rate:.2f}%")
        result_lines.append("")

        # Add pass@k results
        result_lines.append("Pass@k Results:")
        for k_val in k:
            pass_k = pass_k_results.get(f"pass@{k_val}", 0)
            result_lines.append(f"  pass@{k_val}: {pass_k:.4f}")
        result_lines.append("")

        # Add results by difficulty category
        result_lines.append("By Difficulty:")
        result_lines.append(f"  Easy:   {easy_passed}/{easy_total} = {easy_rate:.2f}%")
        result_lines.append(f"  Medium: {medium_passed}/{medium_total} = {medium_rate:.2f}%")
        result_lines.append(f"  Hard:   {hard_passed}/{hard_total} = {hard_rate:.2f}%")
        result_lines.append("")

        # Add overall token statistics
        result_lines.append("Token Statistics (Overall):")
        result_lines.append(f"  Valid sample count: {valid_token_count}")
        result_lines.append(f"  Total Prompt Tokens: {total_prompt_tokens}")
        result_lines.append(f"  Total Completion Tokens: {total_completion_tokens}")
        result_lines.append(f"  Total Tokens: {total_tokens}")
        result_lines.append(f"  Avg Prompt Tokens/sample: {avg_prompt_tokens:.2f}")
        result_lines.append(f"  Avg Completion Tokens/sample: {avg_completion_tokens:.2f}")
        result_lines.append(f"  Avg Total Tokens/sample: {avg_total_tokens:.2f}")
        
        # Add max_completion_tokens statistics
        if max_completion_tokens_list:
            avg_max_completion_tokens = sum(max_completion_tokens_list) / len(max_completion_tokens_list)
            result_lines.append(f"  Max Completion Tokens: {max(max_completion_tokens_list)}")
            result_lines.append(f"  Avg Max Completion Tokens: {avg_max_completion_tokens:.2f}")
        result_lines.append("")

        # Add token statistics by difficulty category
        result_lines.append("Token Statistics (By Difficulty):")
        result_lines.append(f"  Easy:   Avg {easy_avg_tokens:.2f} tokens/sample ({easy_token_count} samples)")
        result_lines.append(f"  Medium: Avg {medium_avg_tokens:.2f} tokens/sample ({medium_token_count} samples)")
        result_lines.append(f"  Hard:   Avg {hard_avg_tokens:.2f} tokens/sample ({hard_token_count} samples)")
        result_lines.append("")

        # Add token statistics by stage (if detailed data available)
        if ilr_count > 0 or code_count > 0:
            result_lines.append("Token Statistics (By Stage):")
            if ilr_count > 0:
                avg_ilr_tokens = ilr_total_tokens / ilr_count
                result_lines.append(f"  ILR Generation: Total {ilr_total_tokens} tokens, Avg {avg_ilr_tokens:.2f} tokens/sample ({ilr_count} samples)")
                result_lines.append(f"    - Prompt: {ilr_prompt_tokens} tokens")
                result_lines.append(f"    - Completion: {ilr_completion_tokens} tokens")
            if code_count > 0:
                avg_cg_tokens = code_total_tokens / code_count
                result_lines.append(f"  Code Generation: Total {code_total_tokens} tokens, Avg {avg_cg_tokens:.2f} tokens/sample ({code_count} samples)")
                result_lines.append(f"    - Prompt: {code_prompt_tokens} tokens")
                result_lines.append(f"    - Completion: {code_completion_tokens} tokens")
            result_lines.append("")
        result_lines.append("="*60)

        # Save to results.txt
        results_txt_path = results_file.parent / "results.txt"
        with open(results_txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(result_lines))

        # Save token statistics to a separate JSON file
        token_stats_path = results_file.parent / "token_stats.json"
        token_stats = {
            "total_tasks": valid_token_count,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "avg_total_tokens": avg_total_tokens,
            "max_completion_tokens_stats": {
                "max_completion_tokens": max(max_completion_tokens_list) if max_completion_tokens_list else 0,
                "avg_max_completion_tokens": sum(max_completion_tokens_list) / len(max_completion_tokens_list) if max_completion_tokens_list else 0
            },
            "by_difficulty": {
                "easy": {
                    "count": easy_token_count,
                    "total_tokens": easy_tokens,
                    "avg_tokens": easy_avg_tokens
                },
                "medium": {
                    "count": medium_token_count,
                    "total_tokens": medium_tokens,
                    "avg_tokens": medium_avg_tokens
                },
                "hard": {
                    "count": hard_token_count,
                    "total_tokens": hard_tokens,
                    "avg_tokens": hard_avg_tokens
                }
            },
            "by_stage": {
                "ilr_generation": {
                    "count": ilr_count,
                    "prompt_tokens": ilr_prompt_tokens,
                    "completion_tokens": ilr_completion_tokens,
                    "total_tokens": ilr_total_tokens,
                    "avg_tokens": ilr_total_tokens / ilr_count if ilr_count > 0 else 0
                },
                "code_generation": {
                    "count": code_count,
                    "prompt_tokens": code_prompt_tokens,
                    "completion_tokens": code_completion_tokens,
                    "total_tokens": code_total_tokens,
                    "avg_tokens": code_total_tokens / code_count if code_count > 0 else 0
                }
            }
        }
        with open(token_stats_path, 'w', encoding='utf-8') as f:
            json.dump(token_stats, f, indent=4, ensure_ascii=False)

        print(f"Evaluation results saved to: {results_txt_path}")
        print(f"Token statistics saved to: {token_stats_path}")

        # Print results to console
        print('\n' + '\n'.join(result_lines))
