"""
Batch Reflection Agent
Used for batch processing of failed test samples, performing three-step analysis and regenerating ILR and code
Supports iterative optimization with up to 3 iterations
"""

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Support direct script execution (python src/agents/batch_reflection_agent.py):
# ensure src/ is on sys.path so utils/tools/agents imports resolve.
_SRC_DIR = Path(__file__).resolve().parent.parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.vision_api_client import VisionAPIClient
from tools.ilr_interpreter import ILRInterpreter, ILRTester
from tools.ocr_extractor import OCRExtractor
from tools.code_tester import CodeTester
from tools.flowchart_cache import FlowchartCache
from tools.problem_extractor import ProblemExtractor
from agents.agent1_ilr_generator_with_tools import ILRGeneratorWithTools
from agents.agent2_code_generator import CodeGenerator
from utils.code_utils import extract_code
from utils.intermediate_representation import (
    get_intermediate_representation_file_name,
    normalize_intermediate_representation_type,
)
from evaluation.evaluate_lcb import build_lcb_input_output, run_test


def str2bool(value):
    """Parse command-line boolean argument, supports true/false."""
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes", "y", "on"}:
        return True
    if normalized in {"false", "0", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse boolean value: {value}")


@dataclass
class FailedSample:
    """Failed sample data structure"""
    task_id: str
    prompt: str
    entry_point: str
    test: str
    test_cases: List[Dict]
    original_ilr: str
    original_code: str
    test_result: Dict
    image_path: str
    starter_code: str = ""  # Added starter_code field
    dataset: str = ""       # Added dataset field
    difficulty: str = "Easy" # Added difficulty field
    problem_text_description: str = ""
    lcb_mode: str = ""
    # Public test cases (sample_io, e.g. the problem-statement examples as
    # assert statements). Ported from the early batch reflection agent: they
    # drive the ERROR CONTEXT given to the LLM only — pass/fail judgment
    # always uses the full (hidden) test suite.
    sample_io: Optional[List[str]] = None
    sample_io_result: Optional[Dict] = None  # public-test result of the current code


@dataclass
class IterationResult:
    """Iteration result data structure"""
    iteration: int
    task_id: str
    issue_type: str  # "ocr", "ilr_logic", "code"
    analysis: str
    regenerated_ilr: Optional[str]
    regenerated_code: Optional[str]
    test_result: Optional[Dict]
    passed: bool
    token_usage: Optional[Dict] = None
    duration: float = 0.0


class BatchReflectionAgent:
    """
    Batch Reflection Agent

    Features:
    1. Batch process failed test samples
    2. Two-step analysis: OCR+ILR logic check -> Code check
    3. Regenerate ILR and code based on issue type
    4. Support up to 3 iterations of optimization
    5. Save results of each iteration
    """

    def __init__(self,
                 config_path: str,
                 results_file: str,
                 data_dir: str,
                 output_dir: str = None,
                 max_iterations: int = 3,
                 reflection_mode: str = "full",
                 intermediate_representation_type: str = "ilr",
                 text_ir_dir: Optional[str] = None,
                 prompt_variant: str = "default",
                 include_problem_text_description: bool = True,
                 allow_ilr_claim_flip: bool = True):
        """
        Initialize the Batch Reflection Agent

        Args:
            config_path: Model config file path
            results_file: Test results file path (samples.jsonl_results.jsonl)
            data_dir: Data directory (containing images)
            output_dir: Output directory, defaults to same directory as results_file
            max_iterations: Maximum number of iterations
            reflection_mode: Reflection mode, full=normal batch reflection, code_only=code ablation
        """
        # Load config
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)

        # Create API client
        self.api_client = VisionAPIClient(config_dict=self.config)

        # Load results data
        self.results_file = Path(results_file)
        self.samples_file = self._derive_samples_file(self.results_file)
        self.samples_data = self._load_samples()
        self.results_data = self._load_results()

        # Data directory
        self.data_dir = Path(data_dir)
        self.dataset_name = self.data_dir.name
        self.reflection_mode = self._normalize_reflection_mode(reflection_mode)
        self.problem_extractor = ProblemExtractor(data_root=str(self.data_dir.parent))
        self.intermediate_representation_type = normalize_intermediate_representation_type(
            intermediate_representation_type
        )
        self.prompt_variant = (prompt_variant or "default").strip().lower()
        self.include_problem_text_description = include_problem_text_description
        # Coverage-gap preservation channel (evidence-gated ILR_LOGIC_ISSUE
        # flip). Keep True to let verifiable ILR claims route back to ILR
        # reflection; a wrong flip costs at most one wasted ILR regeneration
        # (force_code_reflection returns later iterations to code reflection).
        # Set False for strict attribution purity (then a verified-correct ILR
        # is always attributed to the code branch).
        self.allow_ilr_claim_flip = allow_ilr_claim_flip
        # Deterministic ILR interpreter verdict of the most recent gate run
        # (v2 diagnosis gating); read by the prompt builders so the LLM always
        # sees the authoritative test result.
        self._last_ilr_verdict: Optional[Dict[str, Any]] = None
        self.text_ir_map = self._load_text_ir_map(text_ir_dir)
        # Public test cases (sample_io) map: task_id -> list of assert strings.
        # Loaded lazily in get_failed_samples from <data_dir>/*_with_sample_io.jsonl.
        self._sample_io_map: Optional[Dict[str, List[str]]] = None

        # Output directory
        if output_dir is None:
            if self.reflection_mode == "full":
                default_dir_name = "reflection"
            elif self.reflection_mode == "code_only":
                default_dir_name = "reflection_code_only"
            elif self.reflection_mode == "ilr_only":
                default_dir_name = "reflection_ilr_only"
            else:
                default_dir_name = "reflection"
            self.output_dir = self.results_file.parent / default_dir_name
        else:
            self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Maximum number of iterations
        self.max_iterations = max_iterations

        # Initialize tools
        self.code_tester = CodeTester()
        self.ocr_extractor = None
        self.ilr_tester = None
        self.flowchart_cache = None
        self.ilr_generator = None

        if self.reflection_mode in ("full", "ilr_only"):
            self.ocr_extractor = OCRExtractor(vision_model_api=self.api_client)
            self.ilr_tester = ILRTester()

            # OCR cache: use the main output directory (for reading existing OCR cache)
            # Calculate the main output directory: usually output/{dataset}/
            main_output_dir = Path(__file__).parent.parent.parent / "output"
            self.flowchart_cache = FlowchartCache(output_dir=str(main_output_dir))

            # Initialize generator
            # ILRGenerator also uses the main output directory to read OCR cache
            self.ilr_generator = ILRGeneratorWithTools(
                vision_model_api=self.api_client,
                output_dir=str(main_output_dir),  # Use main output directory
                intermediate_representation_type=self.intermediate_representation_type,
                prompt_variant=self.prompt_variant,
                include_problem_text_description=self.include_problem_text_description,
            )

        # All modes need CodeGenerator
        self.code_generator = CodeGenerator(
            config_dict=self.config,
            output_dir=str(self.output_dir),
            prompt_variant=self.prompt_variant,
            include_problem_text_description=self.include_problem_text_description,
        )

    def _normalize_reflection_mode(self, reflection_mode: str) -> str:
        """Normalize reflection mode."""
        normalized = (reflection_mode or "full").strip().lower()
        aliases = {
            "full": "full",
            "normal": "full",
            "default": "full",
            "code_only": "code_only",
            "code-only": "code_only",
            "code": "code_only",
            "ilr_only": "ilr_only",
            "ilr-only": "ilr_only",
            "ilr": "ilr_only",
        }
        if normalized not in aliases:
            raise ValueError(f"Unsupported reflection_mode: {reflection_mode}")
        return aliases[normalized]

    def _use_self_planning(self) -> bool:
        return self.prompt_variant == "self_planning"

    def _maybe_self_planning_block(self) -> str:
        if not self._use_self_planning():
            return ""
        return "Self-Planning (internal): briefly plan the analysis steps and checkpoints. Do NOT output the plan."

    def _build_interface_section(self, starter_code: str, entry_point: str) -> str:
        """Build interface constraints description."""
        sections = []
        if starter_code and starter_code.strip():
            sections.append(
                "Starter Code / Interface:\n"
                f"```python\n{starter_code}\n```"
            )
        elif entry_point:
            sections.append(f"Entry Point: `{entry_point}`")
        return "\n\n".join(sections)

    def _build_problem_text_section(self, sample: FailedSample) -> str:
        """Build problem text description section."""
        if not self.include_problem_text_description:
            return ""

        description = (sample.problem_text_description or "").strip()
        if not description:
            return ""

        return f"""
Problem Text Description:
{description}
"""

    def _parse_response_payload(self, response: Any) -> Optional[Dict[str, Any]]:
        """Try to parse response into a structured dict; return None for pure code strings."""
        if isinstance(response, dict):
            return response
        if not isinstance(response, str):
            return None

        response = response.strip()
        if not response:
            return None

        try:
            parsed = json.loads(response)
        except Exception:
            return None

        return parsed if isinstance(parsed, dict) else None

    def _extract_entry_point_from_starter_code(self, starter_code: str) -> str:
        """Extract entry function name from starter_code."""
        if not starter_code:
            return ""
        match = re.search(r"def\s+(\w+)\s*\(", starter_code)
        return match.group(1) if match else ""

    def _get_problem_record(self, task_id: str, dataset: str) -> Optional[Dict[str, Any]]:
        """Read problem info preferring the specified dataset."""
        if dataset:
            try:
                problem = self.problem_extractor._find_problem_in_dataset(task_id, dataset)
                if problem:
                    return problem
            except Exception:
                pass
        return self.problem_extractor.extract_problem_by_image_name(task_id)

    def _build_problem_info_for_testing(self, sample: FailedSample) -> Dict[str, Any]:
        """Construct problem_info required by CodeTester."""
        problem_info = {
            "prompt": "",
            "test": sample.test,
            "entry_point": self._get_entry_point_for_testing(sample),
            "test_cases": sample.test_cases,
            "starter_code": sample.starter_code,
        }
        if sample.dataset == "LiveCodeBench":
            problem_info.update({
                "lcb_mode": sample.lcb_mode or ("stdio" if sample.entry_point == "solve" else "call_based"),
                "func_name": sample.entry_point,
                "public_test_cases": sample.test_cases,
            })
        return problem_info

    def _test_lcb_sample(self, sample: FailedSample, code: str) -> Dict[str, Any]:
        """Evaluate an LCB sample through the canonical mixed-I/O evaluator."""
        problem_info = self._build_problem_info_for_testing(sample)
        input_output = build_lcb_input_output(problem_info, sample.test_cases)
        if sample.task_id.endswith("_abc363_f"):
            input_output["output_validator"] = "abc363_f"
        if not input_output.get("inputs"):
            return {
                "success": False, "passed": 0, "total": 0, "pass_rate": 0.0,
                "results": [], "error": "No LCB test cases provided",
                "test_method": "lcb_mixed_io",
            }
        raw_results, metadata = run_test(
            {"input_output": json.dumps(input_output, ensure_ascii=False)},
            test=code,
            timeout=6,
        )
        raw_results = raw_results or []
        details = []
        for index, passed in enumerate(raw_results):
            original_case_input = ""
            if index < len(sample.test_cases) and isinstance(sample.test_cases[index], dict):
                original_case_input = sample.test_cases[index].get("input", "")
            details.append({
                "test_num": index + 1,
                "passed": passed is True,
                "input": original_case_input or (input_output["inputs"][index] if index < len(input_output["inputs"]) else ""),
                "expected": input_output["outputs"][index] if index < len(input_output["outputs"]) else "",
                "actual": None,
                "error": None if passed is True else (metadata or {}).get("error_message", "Wrong Answer"),
            })
        passed_count = sum(1 for item in raw_results if item is True)
        total = len(input_output.get("inputs", []))
        return {
            "success": passed_count == total and total > 0,
            "passed": passed_count,
            "total": total,
            "pass_rate": passed_count / total * 100 if total else 0.0,
            "results": details,
            "metadata": metadata or {},
            "test_method": "lcb_mixed_io",
        }
    def _get_entry_point_for_testing(self, sample: FailedSample) -> str:
        """Determine whether to pass entry_point to CodeTester based on test code format."""
        test_code = sample.test or ""

        # HumanEval-style tests define/call check(candidate), need to pass entry_point
        if "check(" in test_code:
            return sample.entry_point

        # LeetCode/Algorithm-style usually directly assert my_solution.xxx(...),
        # if entry_point is forcibly passed, CodeTester will additionally append check(entry_point) causing NameError.
        if test_code:
            return ""

        return sample.entry_point

    def _ensure_detailed_test_result(self, sample: FailedSample) -> Dict[str, Any]:
        """Supplement detailed test results for the sample."""
        test_result = sample.test_result or {}
        if test_result.get('results') and test_result.get('total', 0) > 0:
            return test_result

        if not sample.original_code.strip():
            return test_result

        try:
            if sample.dataset == "LiveCodeBench":
                return self._test_lcb_sample(sample, sample.original_code)
            return self.code_tester.test_single_problem(
                generated_code=sample.original_code,
                problem_info=self._build_problem_info_for_testing(sample)
            )
        except Exception as exc:
            return {
                'success': False,
                'passed': 0,
                'total': 0,
                'results': [],
                'error': f"Failed to supplement test results: {exc}"
            }

    def _load_results(self) -> Dict[str, Any]:
        """Load results data"""
        results = {}
        with open(self.results_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    task_id = data.get('task_id')
                    if task_id:
                        # Evaluation rows only contain pass/test fields; merge the raw sample
                        # to restore completion, response, and ILR context.
                        source = self.samples_data.get(task_id, {})
                        merged = dict(source)
                        merged.update(data)
                        results[task_id] = merged
        return results

    @staticmethod
    def _derive_samples_file(results_file: Path) -> Path:
        """Derive the raw samples.jsonl path from an evaluation result path."""
        name = results_file.name
        if name == "samples.jsonl_results.jsonl":
            return results_file.with_name("samples.jsonl")
        match = re.match(r"^(samples\.jsonl)_results\d+\.jsonl$", name)
        if match:
            return results_file.with_name(match.group(1))
        return results_file.with_name("samples.jsonl")

    def _load_samples(self) -> Dict[str, Dict[str, Any]]:
        """Load the raw samples.jsonl next to the evaluation results."""
        samples: Dict[str, Dict[str, Any]] = {}
        if not self.samples_file.exists():
            return samples
        with self.samples_file.open('r', encoding='utf-8', errors='replace') as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except (TypeError, json.JSONDecodeError):
                    continue
                task_id = data.get('task_id')
                if task_id:
                    samples[task_id] = data
        return samples

    def _load_text_ir_map(self, text_ir_dir: Optional[str]) -> Dict[str, str]:
        """Load text intermediate representation mapping from text_ir.jsonl."""
        if not text_ir_dir:
            return {}

        path = Path(text_ir_dir)
        if path.is_dir():
            file_path = path / get_intermediate_representation_file_name("text")
        else:
            file_path = path

        if not file_path.exists():
            print(f"Warning: Text intermediate representation file not found: {file_path}")
            return {}

        text_ir_map: Dict[str, str] = {}
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except Exception:
                    continue

                if isinstance(item, dict):
                    if "task_id" in item and "text_ir" in item:
                        text_ir_map[item["task_id"]] = item["text_ir"]
                    else:
                        for key, value in item.items():
                            text_ir_map[key] = value

        return text_ir_map

    # ------------------------------------------------------------------
    # Public test cases (sample_io) machinery — ported incrementally from
    # the early batch reflection agent. sample_io drives the ERROR CONTEXT
    # for LLM reflection only; pass/fail is always judged on the full
    # (hidden) test suite, and hidden-test details are withheld from the
    # LLM when the public tests all pass.
    # ------------------------------------------------------------------

    def _load_sample_io_data(self) -> Dict[str, List[str]]:
        """Load public test cases (sample_io) from <data_dir>/*_with_sample_io.jsonl."""
        if self._sample_io_map is not None:
            return self._sample_io_map

        sample_io_map: Dict[str, List[str]] = {}
        sample_io_files = list(self.data_dir.glob('*_with_sample_io.jsonl'))
        if not sample_io_files:
            print(f"Warning: no *_with_sample_io.jsonl found under {self.data_dir}")
            self._sample_io_map = sample_io_map
            return sample_io_map

        sample_io_file = sample_io_files[0]
        print(f"Found sample_io file: {sample_io_file.name}")
        with open(sample_io_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                task_id = data.get('task_id')
                sample_io = data.get('sample_io', [])
                if task_id and sample_io:
                    sample_io_map[task_id] = sample_io
        print(f"Loaded sample_io public test cases for {len(sample_io_map)} tasks")
        self._sample_io_map = sample_io_map
        return sample_io_map

    def _test_with_sample_io(self, code: str, sample_io: List[str], entry_point: str) -> Dict:
        """
        Run the code against the public test cases (sample_io assert statements).

        Each assertion runs inside try/except printing PASS/FAIL/ERROR markers;
        the script executes in a subprocess with a timeout (Windows-safe).
        Result dict is compatible with CodeTester's shape.

        Args:
            code: generated code
            sample_io: public test assert statements
            entry_point: entry function name (placeholder substitution target)

        Returns:
            Test result dict {'success', 'passed', 'total', 'pass_rate', 'results', 'error'}
        """
        if not sample_io:
            return {'success': False, 'passed': 0, 'total': 0, 'pass_rate': 0,
                    'results': [], 'error': 'No sample_io available'}

        cleaned_code = self.code_tester._clean_code(code)

        # Replace placeholder function names (candidate/solution/solve/answer)
        # with the actual entry point — only when an entry point is known.
        normalized_sample_io = []
        placeholder_patterns = ['candidate', 'solution', 'solve', 'answer']
        for assertion in sample_io:
            normalized_assertion = assertion.strip()
            if entry_point:
                for placeholder in placeholder_patterns:
                    pattern = r'\b' + re.escape(placeholder) + r'\s*\('
                    if re.search(pattern, normalized_assertion):
                        normalized_assertion = re.sub(pattern, entry_point + '(', normalized_assertion)
                        break  # first matching placeholder wins, avoid double substitution
            normalized_sample_io.append(normalized_assertion)

        test_script_lines = [cleaned_code, "", "# Sample IO tests", "if __name__ == '__main__':"]
        for i, assertion in enumerate(normalized_sample_io):
            test_script_lines.append("    try:")
            test_script_lines.append(f"        {assertion}")
            test_script_lines.append(f"        print('TEST_{i}_PASSED')")
            test_script_lines.append("    except AssertionError as e:")
            test_script_lines.append(f"        print('TEST_{i}_FAILED: ' + str(e))")
            test_script_lines.append("    except Exception as e:")
            test_script_lines.append(f"        print('TEST_{i}_ERROR: ' + type(e).__name__ + ': ' + str(e))")

        test_script = '\n'.join(test_script_lines)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False,
                                         encoding='utf-8') as f:
            f.write(test_script)
            temp_file = f.name

        try:
            result = subprocess.run(
                [sys.executable, temp_file],
                capture_output=True,
                text=True,
                timeout=10,
                encoding='utf-8',
                errors='replace',
            )
            stdout = result.stdout
            stderr = result.stderr

            results = []
            passed = 0
            for i, assertion in enumerate(normalized_sample_io):
                marker_pass = f'TEST_{i}_PASSED'
                marker_fail = f'TEST_{i}_FAILED'
                marker_error = f'TEST_{i}_ERROR'

                if marker_pass in stdout:
                    results.append({'test_num': i, 'assertion': assertion,
                                    'passed': True, 'error': None})
                    passed += 1
                elif marker_fail in stdout:
                    error_line = ''
                    for line in stdout.split('\n'):
                        if marker_fail in line:
                            error_line = line.split(':', 1)[-1].strip()
                            break
                    results.append({'test_num': i, 'assertion': assertion, 'passed': False,
                                    'error': f'AssertionError: {error_line}' if error_line else 'AssertionError'})
                elif marker_error in stdout:
                    error_line = ''
                    for line in stdout.split('\n'):
                        if marker_error in line:
                            error_line = line.split(':', 1)[-1].strip()
                            break
                    results.append({'test_num': i, 'assertion': assertion, 'passed': False,
                                    'error': error_line or 'Runtime error'})
                else:
                    results.append({'test_num': i, 'assertion': assertion, 'passed': False,
                                    'error': stderr.strip() if stderr else 'Unknown error'})

            return {
                'success': True,
                'passed': passed,
                'total': len(normalized_sample_io),
                'pass_rate': passed / len(normalized_sample_io) * 100 if normalized_sample_io else 0,
                'results': results,
                'error': None
            }

        except subprocess.TimeoutExpired:
            return {'success': False, 'passed': 0, 'total': len(sample_io), 'pass_rate': 0,
                    'results': [], 'error': 'Sample IO test timed out'}
        except Exception as e:
            return {'success': False, 'passed': 0, 'total': len(sample_io), 'pass_rate': 0,
                    'results': [], 'error': f'Sample IO test error: {str(e)}'}
        finally:
            try:
                os.unlink(temp_file)
            except OSError:
                pass

    def _build_error_summary(self, test_result: Dict, source: str = "test") -> str:
        """
        Build an LLM-facing error summary from a test result dict.

        Compatible with both the full test result shape (input/expected/actual)
        and the sample_io shape (assertion/error). Shows at most the first
        failed case.
        """
        total = test_result.get('total', 0)
        passed_count = test_result.get('passed', 0)

        if passed_count >= total > 0:
            return ""

        failed_tests = [r for r in test_result.get('results', []) if not r.get('passed', False)]
        source_label = "Public test (sample_io)" if source == "sample_io" else "Hidden test"
        error_summary = f"Code failed {len(failed_tests)}/{total} {source_label} cases\n"

        for i, test in enumerate(failed_tests[:1], 1):
            if source == "sample_io":
                assertion = test.get('assertion', 'N/A')
                error_msg = test.get('error', 'Unknown error')
                error_summary += f"  Test {i}:\n"
                error_summary += f"    Assertion: {assertion}\n"
                error_summary += f"    Error: {error_msg}\n"
            else:
                test_input = test.get('input', test.get('input_args', 'N/A'))
                expected = test.get('expected_output', test.get('expected', 'N/A'))
                actual = test.get('actual_output', test.get('actual', 'N/A'))
                error_msg = test.get('error', 'Unknown error')
                error_summary += f"  Test {i}:\n"
                error_summary += f"    Input: {test_input}\n"
                error_summary += f"    Expected: {expected}\n"
                error_summary += f"    Actual: {actual}\n"
                error_summary += f"    Error: {error_msg}\n"

        return error_summary

    def _is_public_context_withheld(self, sample: FailedSample) -> bool:
        """True when sample_io exists, was tested, and the current code passes
        ALL public cases — the LLM context then withholds hidden-test details."""
        r = sample.sample_io_result
        return bool(r and r.get('total', 0) > 0 and r.get('passed', 0) == r.get('total', 0))

    def _get_error_context_for_llm(self, sample: FailedSample) -> str:
        """
        Error context for LLM reflection, public-test-first:
          1. sample_io failures -> first failing public assertion;
          2. sample_io all passed -> generic hint, hidden-test details withheld;
          3. no sample_io -> fall back to the full test result (legacy behavior).
        """
        # 1. prefer public test failures
        if sample.sample_io_result is not None and sample.sample_io_result.get('total', 0) > 0:
            sample_io_total = sample.sample_io_result.get('total', 0)
            sample_io_passed = sample.sample_io_result.get('passed', 0)
            if sample_io_passed < sample_io_total:
                return self._build_error_summary(sample.sample_io_result, "sample_io")
            # 2. all public tests pass: do not leak hidden test details, but
            # DO show the public cases — they are the LLM's leak-free ground
            # truth for the problem semantics (guards against hallucinated
            # counterexamples when the concrete failure details are withheld)
            public_lines = "\n".join(f"  {a}" for a in (sample.sample_io or []))
            return ("The code passes all public test cases (sample_io), but fails some "
                    "hidden test cases. Please review the code logic for edge cases.\n"
                    "Public test cases (your ground truth for the problem semantics):\n"
                    f"{public_lines}")
        # 3. fallback: full test result (legacy behavior when no sample_io)
        test_result = sample.test_result
        if test_result and test_result.get('total', 0) > 0:
            return self._build_error_summary(test_result, "test")
        return ""

    def get_failed_samples(self) -> List[FailedSample]:
        """Get all failed samples"""
        failed_samples = []

        # Load public test cases (sample_io) for error-context reflection
        sample_io_map = self._load_sample_io_data()

        # Debug info
        total_records = len(self.results_data)
        passed_count = 0
        failed_count = 0
        skipped_count = 0
        no_sample_io_count = 0

        for task_id, data in self.results_data.items():
            # Check if passed
            # Note: if passed field doesn't exist, default to False instead of True, because we only care about failures
            passed = data.get('passed', False)
            if passed:
                passed_count += 1
                continue

            failed_count += 1

            response_payload = self._parse_response_payload(data.get('response'))
            response_problem_info = response_payload.get('problem_info', {}) if response_payload else {}
            # Determine dataset name from the record itself; do NOT fall back to
            # self.dataset_name (data_dir.name) because that can misidentify
            # non-LCB samples as LiveCodeBench, routing test_cases retrieval
            # into the wrong branch and producing an empty list (0/0 bug).
            dataset_name = (
                data.get('dataset')
                or response_problem_info.get('dataset')
                or ""
            )
            problem_record = self._get_problem_record(task_id, dataset_name) or {}

            starter_code = (
                response_problem_info.get('starter_code')
                or problem_record.get('starter_code')
                or data.get('starter_code', '')
            )
            problem_text_description = (
                response_problem_info.get('text_description')
                or problem_record.get('text_description')
                or ""
            )
            entry_point = (
                response_problem_info.get('entry_point')
                or problem_record.get('entry_point')
                or self._extract_entry_point_from_starter_code(starter_code)
            )
            lcb_mode = (
                data.get("lcb_mode")
                or response_problem_info.get("lcb_mode")
                or problem_record.get("lcb_mode")
                or ("stdio" if entry_point == "solve" else "call_based")
            ) if dataset_name == "LiveCodeBench" else ""
            # Unified test_cases retrieval: always prefer test_cases first,
            # then fall back to public_test_cases.  Do NOT branch on dataset
            # to avoid picking up an empty public_test_cases list when
            # test_cases is the populated field (the 0/0 bug).
            test_cases = (
                response_problem_info.get("test_cases")
                or problem_record.get("test_cases")
                or response_problem_info.get("public_test_cases")
                or problem_record.get("public_test_cases")
                or []
            )
            test_code = data.get('test') or response_problem_info.get('test') or ""
            difficulty = (
                data.get('difficulty')
                or response_problem_info.get('difficulty')
                or problem_record.get('difficulty')
                or data.get('meta', {}).get('difficulty')
                or 'Easy'
            )
            generated_code = (
                (response_payload or {}).get('generated_code')
                or (response_payload or {}).get('completion')
                or (
                    data.get('response', '')
                    if isinstance(data.get('response'), str)
                    else ""
                )
                or data.get('completion', '')
            )
            cleaned_generated_code = extract_code(generated_code) if isinstance(generated_code, str) else generated_code
            original_ilr = (
                (response_payload or {}).get('ilr')
                or (response_payload or {}).get('intermediate_representation')
                or ""
            )
            if self.text_ir_map and task_id in self.text_ir_map:
                if self.intermediate_representation_type == "text" or not original_ilr:
                    original_ilr = self.text_ir_map[task_id]
            test_result = (response_payload or {}).get('test_result') or data.get('test_result') or {}

            # Find image path
            image_path = self._find_image_path(task_id)
            if not image_path:
                print(f"Warning: Cannot find image file {task_id}")
                skipped_count += 1
                continue

            failed_sample = FailedSample(
                task_id=task_id,
                prompt="",
                entry_point=entry_point,
                test=test_code,
                test_cases=test_cases,
                original_ilr=original_ilr,
                original_code=cleaned_generated_code,
                test_result=test_result,
                image_path=str(image_path),
                starter_code=starter_code,
                dataset=dataset_name,
                difficulty=difficulty,
                problem_text_description=problem_text_description,
                lcb_mode=lcb_mode,
                sample_io=sample_io_map.get(task_id, []),
            )
            if not sample_io_map.get(task_id):
                no_sample_io_count += 1
            print(f"  [Debug] {task_id}: dataset={dataset_name!r}, test_cases={len(test_cases)}, entry_point={entry_point!r}")
            failed_sample.test_result = self._ensure_detailed_test_result(failed_sample)

            failed_samples.append(failed_sample)

        print(f"\n[Debug Info] Total records: {total_records}")
        print(f"[Debug Info] Passed: {passed_count}")
        print(f"[Debug Info] Failed: {failed_count}")
        print(f"[Debug Info] Skipped (parse error / no image): {skipped_count}")
        print(f"[Debug Info] Without sample_io: {no_sample_io_count}")
        print(f"[Debug Info] Final collected failed samples: {len(failed_samples)}")

        return failed_samples

    def _find_image_path(self, task_id: str) -> Optional[Path]:
        """Find image file path"""
        # Try multiple possible paths
        possible_paths = [
            self.data_dir / 'images' / f'{task_id}.png',
            self.data_dir / 'images' / f'{task_id}.jpg',
            self.data_dir / 'images' / f'{task_id}.jpeg',
        ]

        for path in possible_paths:
            if path.exists():
                return path

        return None

    def analyze_sample(self, sample: FailedSample) -> Tuple[str, str, Dict, Optional[Dict]]:
        """
        Three-step analysis of failed sample

        Returns:
            (issue_type, analysis, usage, ocr_data) - Issue type, analysis result, token usage and OCR data
            issue_type: "ocr", "ilr_logic", "code"
        """
        print(f"\n{'='*60}")
        print(f"Analyzing sample: {sample.task_id}")
        print(f"{'='*60}")

        if self.reflection_mode == "code_only":
            analysis, usage = self._analyze_code_only_ablation(
                sample=sample,
                code=sample.original_code,
                test_result=sample.test_result
            )
            return "code", analysis, usage, None

        if self.reflection_mode == "ilr_only":
            # ilr_only mode: only analyze ILR issues, ignore code
            analysis, usage, ocr_data = self._analyze_ilr_only(sample)
            return "ilr", analysis, usage, ocr_data

        if self.intermediate_representation_type == "text":
            failed_tests = [r for r in sample.test_result.get('results', []) if not r.get('passed', False)]
            error_summary = f"Failed {len(failed_tests)}/{sample.test_result.get('total', 0)} test cases\n"
            for i, test in enumerate(failed_tests[:1], 1):
                test_input = test.get('input', test.get('input_args', 'N/A'))
                expected = test.get('expected_output', test.get('expected', 'N/A'))
                actual = test.get('actual_output', test.get('actual', 'N/A'))
                error_msg = test.get('error', 'Unknown error')
                error_summary += f"  Test {i}:\n"
                error_summary += f"    Input: {test_input}\n"
                error_summary += f"    Expected: {expected}\n"
                error_summary += f"    Actual: {actual}\n"
                error_summary += f"    Error: {error_msg}\n"
            analysis, usage = self._get_llm_analysis(
                sample=sample,
                issue_type="ilr",
                context={"error_summary": error_summary},
                current_ilr=sample.original_ilr,
            )
            return "ilr", analysis, usage, None

        total_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
        ocr_data_cache = None  # Used to cache OCR data

        def add_usage(u):
            if u:
                total_usage['prompt_tokens'] += u.get('prompt_tokens', 0)
                total_usage['completion_tokens'] += u.get('completion_tokens', 0)
                total_usage['total_tokens'] += u.get('total_tokens', 0)

        # Step 0 (v2): deterministic ILR interpreter test as the PRIMARY gate.
        # When the ILR reproduces the expected behaviour on every test case,
        # attribute the failure to the code directly: the noisy OCR label
        # comparison must not divert a verified-correct ILR into ILR
        # regeneration (it misattributed 45/50 verified-correct ILRs in the
        # controlled evaluation).
        print("\n[Step 0] Running deterministic ILR interpreter test (primary gate)...")
        ilr_test_result = self.ilr_tester.test_ilr(
            ilr_json=sample.original_ilr,
            test_cases=sample.test_cases,
            entry_point=sample.entry_point,
        )
        ilr_passed = ilr_test_result.get('passed', 0)
        ilr_total = ilr_test_result.get('total', 0)
        ilr_pass_all = ilr_total > 0 and ilr_passed == ilr_total
        self._last_ilr_verdict = {'passed': ilr_passed, 'total': ilr_total,
                                  'pass_all': ilr_pass_all}
        print(f"  > ILR interpreter verdict: {ilr_passed}/{ilr_total} passed")

        if ilr_pass_all:
            print("  > ILR verified correct on all test cases -> code analysis branch...")
            code_analysis, code_usage = self._force_code_analysis(sample)
            add_usage(code_usage)
            # Coverage-gap preservation (evidence-gated): the ILR passed the
            # interpreter, but the tests may not cover every behaviour. Flip
            # back to the ILR branch ONLY when the LLM names a specific ILR
            # node AND provides a (input, ilr_output, expected) counterexample
            # that survives deterministic refutation. Residual risk: an
            # honestly-traced claim built on a WRONG understanding of the
            # problem semantics is unverifiable and may still flip; the cost
            # is bounded (one wasted ILR regeneration, then force_code_
            # reflection returns the loop to code reflection). Disable via
            # allow_ilr_claim_flip=False for strict attribution purity.
            if self.allow_ilr_claim_flip and self._ilr_claim_survives_refutation(sample, code_analysis):
                print("  > LLM provided verifiable evidence of an ILR logic bug -> ILR branch")
                return "ilr", code_analysis, total_usage, ocr_data_cache
            return "code", code_analysis, total_usage, ocr_data_cache

        # The interpreter proved the ILR logic wrong. The OCR comparison and
        # the LLM now serve LOCALIZATION (flowchart text extraction vs logic
        # extraction), not detection.
        print("\n[Step 1] ILR logic error confirmed - checking OCR extraction for localization...")
        ilr_issue, ilr_analysis, ilr_usage, ocr_data_cache = self._check_ocr_and_ilr(
            sample, ilr_test_result=ilr_test_result)
        add_usage(ilr_usage)

        if ilr_issue:
            return "ilr", ilr_analysis, total_usage, ocr_data_cache

        # If LLM determines ILR has no substantive issues, directly switch to code issue analysis
        if "OCR and ILR logic check passed" in ilr_analysis:
            print("  > ILR has no issues (LLM override), automatically switching to code reflection analysis...")
            code_analysis, code_usage = self._force_code_analysis(sample)
            add_usage(code_usage)
            return "code", code_analysis, total_usage, ocr_data_cache

        # Step 2: Check code generation
        print("\n[Step 2] Checking code generation...")
        code_issue, code_analysis, code_usage = self._check_code(sample)
        add_usage(code_usage)

        if code_issue:
            return "code", code_analysis, total_usage, ocr_data_cache

        return "unknown", "Unable to determine issue type", total_usage, ocr_data_cache

    def _analyze_code_only_ablation(self,
                                    sample: FailedSample,
                                    code: str,
                                    test_result: Dict[str, Any]) -> Tuple[str, Dict]:
        """Code ablation mode: only analyze code issues."""
        if test_result.get('passed', 0) >= test_result.get('total', 0) and test_result.get('total', 0) > 0:
            return "Code generation appears normal", {}

        failed_tests = [r for r in test_result.get('results', []) if not r.get('passed', False)]
        error_summary = f"Code generation has issues. Failed {len(failed_tests)}/{test_result.get('total', 0)} test cases\n"
        for i, test in enumerate(failed_tests[:1], 1):
            test_input = test.get('input', test.get('input_args', 'N/A'))
            expected = test.get('expected_output', test.get('expected', 'N/A'))
            actual = test.get('actual_output', test.get('actual', 'N/A'))
            error_msg = test.get('error', 'Unknown error')
            error_summary += f"  Test {i}:\n"
            error_summary += f"    Input: {test_input}\n"
            error_summary += f"    Expected: {expected}\n"
            error_summary += f"    Actual: {actual}\n"
            error_summary += f"    Error: {error_msg}\n"

        context = {
            'error_summary': error_summary,
            'code_only': True,
        }

        first_failed_test = self._get_first_failed_test(test_result)
        if first_failed_test:
            execution_info = self._run_instrumented_failed_case(sample, code, first_failed_test)
            context['runtime_trace'] = self._format_runtime_trace(execution_info)

        return self._get_llm_analysis(
            sample=sample,
            issue_type="code",
            context=context,
            current_ilr=None,
            current_code=code
        )

    def _analyze_ilr_only(self, sample: FailedSample) -> Tuple[str, Dict, Optional[Dict]]:
        """ILR ablation mode: only analyze ILR issues (OCR+logic), not code."""
        if not self.ocr_extractor or not self.ilr_tester or not self.flowchart_cache:
            return "ILR checks are disabled", {}, None

        total_usage = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

        def add_usage(u):
            if u:
                total_usage['prompt_tokens'] += u.get('prompt_tokens', 0)
                total_usage['completion_tokens'] += u.get('completion_tokens', 0)
                total_usage['total_tokens'] += u.get('total_tokens', 0)

        # Only do OCR and ILR logic check
        print("  > ILR Only mode: only checking OCR and ILR logic...")
        ilr_issue, ilr_analysis, ilr_usage, ocr_data = self._check_ocr_and_ilr(sample)
        add_usage(ilr_usage)

        if ilr_issue:
            return ilr_analysis, total_usage, ocr_data

        # If ILR check passes, force returning an ILR analysis (let it regenerate ILR)
        if "OCR and ILR logic check passed" in ilr_analysis:
            # Even if check passes, provide test failure info for LLM to re-examine ILR
            failed_tests = [r for r in sample.test_result.get('results', []) if not r.get('passed', False)]
            error_summary = f"ILR logic test shows failures. Failed {len(failed_tests)}/{sample.test_result.get('total', 0)} test cases\n"
            for i, test in enumerate(failed_tests[:1], 1):
                test_input = test.get('input', test.get('input_args', 'N/A'))
                expected = test.get('expected_output', test.get('expected', 'N/A'))
                actual = test.get('actual_output', test.get('actual', 'N/A'))
                error_msg = test.get('error', 'Unknown error')
                error_summary += f"  Test {i}:\n"
                error_summary += f"    Input: {test_input}\n"
                error_summary += f"    Expected: {expected}\n"
                error_summary += f"    Actual: {actual}\n"
                error_summary += f"    Error: {error_msg}\n"

            context = {'error_summary': error_summary}
            analysis, usage = self._get_llm_analysis(
                sample=sample,
                issue_type="ilr",
                context=context,
                current_ilr=sample.original_ilr,
            )
            add_usage(usage)
            return analysis, total_usage, ocr_data

        return ilr_analysis, total_usage, ocr_data

    @staticmethod
    def _truncate_degenerate_text(text: str, max_chars: int = 16000,
                                  min_line_len: int = 60) -> str:
        """Cut the repetition loops small models fall into before the analysis
        text is stored or fed into follow-up prompts (ILR/code regeneration).

        Degeneration looks like the same list items re-emitted verbatim with
        new numbering ("Bug 9 ... Bug 19 ... Bug 27 ..." with identical
        bodies).  Detect a long line whose numbering-stripped form already
        appeared and cut there, keeping the distinct content; also cap the
        total length as a fallback so prompts stay within the context window.
        """
        if not text:
            return text
        lines = text.split('\n')
        seen = set()
        for i, line in enumerate(lines):
            # Strip list numbering / "Bug N" prefixes so re-emitted items
            # with new numbers still compare equal.
            normalized = re.sub(
                r'^\s*(?:\d+[.)]|[-*•])\s*(?:Bug\s*\d+\s*[:.)]\s*)?',
                '', line).strip()
            if len(normalized) < min_line_len:
                continue
            if normalized in seen:
                print(f"  > Degenerate repetition detected at line {i + 1}; truncating analysis")
                text = '\n'.join(lines[:i]) + \
                    '\n[... truncated: repetitive content in model output ...]'
                break
            seen.add(normalized)
        if len(text) > max_chars:
            text = text[:max_chars] + '\n[... truncated: length limit ...]'
        return text

    def _get_llm_analysis(self, sample: FailedSample, issue_type: str, context: Dict[str, Any],
                          current_ilr: Optional[str] = None,
                          current_code: Optional[str] = None) -> Tuple[str, Dict]:
        """
        Have the LLM generate modification suggestions

        Returns:
            (Analysis suggestions, token usage info)
        """
        print(f"Requesting LLM to generate {issue_type} analysis suggestions...")

        prompt = ""
        if issue_type == "ilr":
            ilr_for_prompt = current_ilr if current_ilr is not None else sample.original_ilr
            # Build OCR mismatch section (if any)
            ocr_section = ""
            if context.get('mismatches'):
                ocr_section = f"""
## OCR Label Mismatch Hints (localization aid only)

ILR labels whose vocabulary is not covered by the OCR-extracted flowchart text:
{context.get('mismatches', '')}

OCR Extracted Text:
{context.get('ocr_text', '')}
"""

            # Build ILR logic error section (if any)
            logic_section = ""
            if context.get('error_summary'):
                logic_section = f"""
## ILR Logic Test Failures

{context.get('error_summary', '')}
"""

            # The deterministic interpreter verdict is authoritative context.
            verdict = self._last_ilr_verdict or {}
            vp, vt = verdict.get('passed', '?'), verdict.get('total', '?')
            prompt = f"""
You are an expert in flowchart analysis, text extraction, and algorithm logic.
A flowchart was converted into an ILR (Intermediate Logic Representation), and
the ILR was executed by a deterministic interpreter against the problem's test cases.

## Deterministic ILR Interpreter Test
Result: the ILR passed {vp}/{vt} test cases.
The interpreter executed the ILR logic directly and compared outputs with the
expected results, so a failing test means the ILR logic bug is real. Do not
question the test result.

Task ID: {sample.task_id}
Problem Description: {sample.prompt}

Current ILR:
{ilr_for_prompt}
{ocr_section}{logic_section}
Analyze WHERE the logic error comes from and how to fix it:
1. OCR/text extraction issues: whether ILR node labels or actions contradict the actual flowchart content (use the OCR mismatch hints above, if any).
2. Logic extraction issues: wrong condition, wrong variable update, wrong branch wiring, missing initialization, swapped arguments.

If — and only if — you believe the ILR logic itself is actually correct and the
failures are caused by interpreter limitations rather than the ILR (rare),
reply with exactly "NO_ISSUE".
Otherwise, provide specific suggestions to fix the ILR.
Focus on fixing control flow, conditions, and variable operations.

IMPORTANT: Keep your response concise (maximum 500 words). Structure your response as:
1. Brief diagnosis: trace through the first failing test case to identify where the logic goes wrong (2-3 sentences)
2. Numbered bug list: each bug with a specific fix suggestion and whether it originated from OCR extraction or logic extraction
Do NOT re-derive the problem, do NOT include mathematical proofs, do NOT generate code.
Do NOT generate the full corrected ILR yet.
"""

        elif issue_type == "code":
            ilr_for_prompt = current_ilr if current_ilr is not None else sample.original_ilr
            code_for_prompt = current_code if current_code is not None else sample.original_code

            # Build instrumented runtime trace section (if any)
            runtime_section = ""
            if context.get('runtime_trace'):
                runtime_section = f"""

Instrumented Execution Details (First Failed Test):
{context.get('runtime_trace', '')}
"""

            # The deterministic interpreter verdict is authoritative context:
            # it tells the LLM whether the ILR itself was verified correct.
            verdict = self._last_ilr_verdict or {}
            vp, vt = verdict.get('passed', '?'), verdict.get('total', '?')
            prompt = f"""
You are an expert Python programmer.
The ILR was verified by a deterministic ILR interpreter: it passed {vp}/{vt} test cases.
The generated Python code (translated from the ILR) failed the test cases, which
implies the code implementation details are wrong (or, rarely, the ILR has a
logic error that the interpreter tests did not cover).

Task ID: {sample.task_id}
Problem Description: {sample.prompt}

Current ILR:
{ilr_for_prompt}

Generated Code:
{code_for_prompt}

Test Failures:
{context.get('error_summary', '')}
{runtime_section}
Please analyze the code and test failures.

If — and only if — you can identify a specific ILR node that is logically wrong
despite the interpreter passing the current tests, emit a dedicated marker line
in EXACTLY this format so your claim can be verified by execution:
ILR_LOGIC_ISSUE: node <id> | input: <python literal> | ilr_output: <python literal> | expected: <python literal>
where `input` is a concrete input that exposes the error, `ilr_output` is your
own hand-trace of what the ILR produces on that input (verified by execution —
a wrong trace invalidates the claim), and `expected` is what the problem
statement and the public test cases imply the correct output should be.
Claims without this verifiable evidence are rejected. Do NOT use the marker for
naming/style differences or unproven suspicions; a wrong claim triggers an
unnecessary ILR regeneration.
Otherwise, treat the failure as a code bug.

IMPORTANT: Keep your response concise (maximum 500 words). Structure your response as:
1. Brief diagnosis: trace through the failing test case to identify where the code deviates from the ILR logic (2-3 sentences)
2. Numbered bug list: each bug with a specific fix suggestion
Do NOT re-derive the problem, do NOT include mathematical proofs, do NOT generate code.
"""

        try:
            # Call LLM to generate analysis
            result = self.api_client.call_api(
                prompt=prompt,
                image_path=sample.image_path if issue_type == "ilr" else None
            )
            analysis = result.get('content', '')
            # Small models can degenerate into repetition loops (identical
            # items re-emitted with new numbering) that bloat follow-up
            # prompts past the context window; cut that before it spreads.
            analysis = self._truncate_degenerate_text(analysis)
            usage = result.get('usage', {})

            return analysis, usage
        except Exception as e:
            print(f"Failed to get LLM analysis: {e}")
            return f"Auto-analysis failed: {str(e)}. Context: {str(context)}", {}

    def _ilr_claim_survives_refutation(self, sample: FailedSample, analysis: str) -> bool:
        """Evidence-gated override used ONLY when the ILR passed all
        interpreter tests (coverage-gap preservation channel).

        The code-branch LLM may claim a specific ILR node is logically wrong
        by emitting an "ILR_LOGIC_ISSUE" marker that MUST carry verifiable
        evidence:
            ILR_LOGIC_ISSUE: node <id> | input: <literal> | ilr_output: <literal> | expected: <literal>
        where `ilr_output` is the LLM's own hand-trace of what the ILR
        produces on `input`, and `expected` is what the problem statement
        implies the correct output should be.

        Deterministic arbiters (no extra LLM calls):
          1. claimed input covered by the (passed) test suite -> refuted;
          2. hand-trace check: ILR actually executes `input` to something
             different from the claimed `ilr_output` -> the LLM misread the
             ILR -> refuted;
          3. ILR output equals the claimed `expected` -> by the LLM's own
             standard the ILR is right on that input -> refuted.
        A claim that survives (honest trace + real divergence + expectation
        off-test) flips the attribution; the reflection loop re-validates
        regenerated ILRs, so a wrong flip only costs tokens.
        """
        if not analysis or "ILR_LOGIC_ISSUE" not in analysis:
            return False

        import ast

        def _try_parse(text):
            text = text.strip().strip('`').strip()
            try:
                return ast.literal_eval(text)
            except Exception:
                return None

        def _extract_field(line, field):
            if f'{field}:' not in line:
                return None, False
            tail = line.split(f'{field}:', 1)[1]
            for opener, closer in (('[', ']'), ('(', ')'), ('{', '}')):
                start = tail.find(opener)
                if start != -1:
                    end = tail.find(closer, start)
                    if end != -1:
                        value = _try_parse(tail[start:end + 1])
                        if value is not None:
                            return value, True
            # fallback: up to the next '|' separator or end of line
            value = _try_parse(tail.split('|')[0])
            return value, value is not None

        for line in analysis.split('\n'):
            if "ILR_LOGIC_ISSUE" not in line:
                continue
            claimed_input, ok_in = _extract_field(line, 'input')
            claimed_ilr_output, ok_out = _extract_field(line, 'ilr_output')
            claimed_expected, ok_exp = _extract_field(line, 'expected')

            if not (ok_in and ok_out and ok_exp):
                print("  > ILR_LOGIC_ISSUE claim lacks parseable "
                      "input/ilr_output/expected evidence -> refuted")
                continue  # this claim is refuted; check further marker lines

            # Arbiter 1: claimed input covered (and passed) by the test suite
            refuted = False
            for tc in sample.test_cases or []:
                args = (tc.get('input') or {}).get('args')
                if args is None:
                    continue
                if (len(args) == 1 and args[0] == claimed_input) or list(args) == (
                        claimed_input if isinstance(claimed_input, list) else [claimed_input]):
                    print(f"  > ILR_LOGIC_ISSUE input {claimed_input!r} is in the test suite "
                          f"which the ILR passed -> refuted")
                    refuted = True
                    break
            if refuted:
                continue

            # Execute the ILR once on the claimed input for arbiters 2 and 3
            try:
                self.ilr_tester.interpreter.parse(sample.original_ilr)
                ctx_inputs = self.ilr_tester._parse_input(
                    {'args': [claimed_input]}, sample.entry_point)
                actual = self.ilr_tester.interpreter.execute(ctx_inputs)
            except Exception as e:
                print(f"  > Could not execute ILR on claimed input ({e}) -> claim survives")
                return True

            # Arbiter 2: the LLM's hand-trace of the ILR must match reality
            if not self.ilr_tester._compare_output(actual, claimed_ilr_output):
                print(f"  > ILR executes claimed input {claimed_input!r} to {actual!r} "
                      f"but the claim traced ilr_output={claimed_ilr_output!r} "
                      f"(misread ILR) -> refuted")
                continue

            # Arbiter 3: divergence from the claimed expected must exist
            if self.ilr_tester._compare_output(actual, claimed_expected):
                print(f"  > ILR executes claimed input {claimed_input!r} to {actual!r} "
                      f"(= claimed expected) -> no divergence, claim refuted")
                continue

            print(f"  > Claim survives: honest trace (ilr_output={claimed_ilr_output!r}), "
                  f"real divergence (expected {claimed_expected!r}), input off-test "
                  f"{claimed_input!r} -> flip to ILR branch")
            return True

        return False

    def _check_ocr_and_ilr(self, sample: FailedSample,
                           ilr_test_result: Optional[Dict[str, Any]] = None
                           ) -> Tuple[bool, str, Dict, Optional[Dict]]:
        """
        Combined check of OCR text extraction and ILR logic correctness

        Returns:
            (has_issue, analysis, usage, ocr_data) - Whether there is an issue, analysis result, token usage, OCR data
        """
        if self.intermediate_representation_type == "text":
            return False, "Text IR checks are skipped in text mode", {}, None
        if not self.ocr_extractor or not self.ilr_tester or not self.flowchart_cache:
            return False, "OCR/ILR checks are disabled in code-only mode", {}, None

        ocr_data = None
        mismatches = []
        ocr_texts = []
        error_summary = ""
        has_ocr_mismatch = False
        has_ilr_logic_error = False

        # Part 1: OCR extraction and comparison
        try:
            # Extract OCR - prefer using cache
            if sample.dataset and self.flowchart_cache.is_extracted(sample.task_id, sample.dataset):
                print(f"  > Using OCR cache: {sample.task_id}")
                ocr_data = self.flowchart_cache.get_flowchart_data(sample.task_id, sample.dataset)
            else:
                print(f"  > Extracting OCR data: {sample.task_id}")
                ocr_data, _ = self.ocr_extractor.extract_raw_text(
                    image_path=sample.image_path,
                    task_id=sample.task_id,
                    dataset=sample.dataset
                )
                if sample.dataset:
                    self.flowchart_cache.save_flowchart_data(sample.task_id, sample.dataset, ocr_data)
                    print(f"  > OCR data cached: {sample.task_id}")

            # Parse ILR (compatible with ```json code blocks)
            try:
                ilr_dict = self._parse_ilr_json(sample.original_ilr)
            except Exception as e:
                print(f"  > ILR parse error (non-OCR): {e}")
                ilr_dict = {}

            # Extract labels from ILR (excluding initialization nodes)
            ilr_labels = []
            for node in ilr_dict.get('nodes', []):
                node_type = node.get('type', '')
                label = node.get('label', '').strip()
                action = node.get('action', '').strip()
                logic = node.get('logic', '').strip()

                # Skip Start nodes and initialization nodes
                if label in ['Start', 'Initialize all variables']:
                    continue

                # Skip obvious initialization actions (containing assignment and at the beginning)
                if action and self._is_initialization_action(action):
                    continue

                if label:
                    ilr_labels.append(label)
                if action and not self._is_initialization_action(action):
                    ilr_labels.append(action)
                if logic:
                    ilr_labels.append(logic)

            # Extract OCR text
            for node in ocr_data.get('nodes', []):
                text = node.get('raw_text', '').strip()
                if text:
                    ocr_texts.append(text)

            # Compare OCR and ILR
            for ilr_label in ilr_labels:
                found = False
                for ocr_text in ocr_texts:
                    if self._text_similarity(ilr_label, ocr_text) > 0.6:
                        found = True
                        break
                if not found and len(ilr_label) > 3:  # Ignore labels that are too short
                    mismatches.append(ilr_label)

            has_ocr_mismatch = len(mismatches) > 0

        except Exception as e:
            print(f"  > OCR check error: {e}")

        # Part 2: ILR logic test
        try:
            if ilr_test_result is None:
                print(f"  > Testing ILR logic with ILRTester... (test_cases={len(sample.test_cases)}, dataset={sample.dataset})")
                # Unified ILR test call: always use test_ilr, do not branch on
                # dataset.  test_lcb internally normalizes to test_ilr anyway,
                # and the branch causes confusion when dataset is misidentified.
                ilr_test_result = self.ilr_tester.test_ilr(
                    ilr_json=sample.original_ilr,
                    test_cases=sample.test_cases,
                    entry_point=sample.entry_point,
                )
            else:
                print("  > Reusing deterministic ILR interpreter verdict from the primary gate")
            passed = ilr_test_result.get('passed', 0)
            total = ilr_test_result.get('total', 0)
            self._last_ilr_verdict = {'passed': passed, 'total': total,
                                      'pass_all': total > 0 and passed == total}
            print(f"  > ILR test result: {passed}/{total} passed")

            if total == 0:
                # ILR parse failure or no test cases – treat as ILR logic error
                has_ilr_logic_error = True
                ilr_error = ilr_test_result.get('error', '')
                if ilr_error:
                    error_summary = f"ILR logic test could not run: {ilr_error}\n"
                else:
                    error_summary = "ILR logic test could not run: no test cases available (total=0)\n"
            elif passed < total:
                has_ilr_logic_error = True
                failed_tests = [r for r in ilr_test_result.get('results', []) if not r.get('passed', False)]
                error_summary = f"Failed {len(failed_tests)}/{total} test cases\n"
                for i, test in enumerate(failed_tests[:1], 1):  # Only show 1 to avoid data leakage
                    test_input = test.get('input', 'N/A')
                    expected = test.get('expected_output', 'N/A')
                    actual = test.get('actual_output', 'N/A')
                    error_msg = test.get('error', 'Unknown error')
                    error_summary += f"  Test {i}:\n"
                    error_summary += f"    Input: {test_input}\n"
                    error_summary += f"    Expected: {expected}\n"
                    error_summary += f"    Actual: {actual}\n"
                    error_summary += f"    Error: {error_msg}\n"

        except Exception as e:
            print(f"  > ILR logic check error: {e}")

        # If both checks pass, return directly
        if not has_ocr_mismatch and not has_ilr_logic_error:
            return False, "OCR and ILR logic checks both normal", {}, ocr_data

        # Build comprehensive context. Design note (public-test reflection vs
        # gate quality): the CODE branch keeps the early sample_io stance
        # (public-test-first, hidden details withheld — see
        # _get_error_context_for_llm). The ILR branch deliberately keeps the
        # interpreter's concrete failing case: the ILR logic error is already
        # a deterministic fact, localization requires the failing trace, the
        # leak is bounded to the first case, and any regenerated ILR is
        # re-validated on the full test suite.
        context = {}
        if has_ocr_mismatch:
            context['ocr_text'] = '\n'.join(ocr_texts)
            context['mismatches'] = '\n'.join(mismatches)
        if has_ilr_logic_error:
            context['error_summary'] = error_summary

        # Get comprehensive LLM analysis
        analysis, usage = self._get_llm_analysis(sample, "ilr", context)

        if "NO_ISSUE" in analysis:
            print("  > LLM determines OCR and ILR logic have no substantive issues, skipping...")
            return False, "OCR and ILR logic check passed (LLM verified)", usage, ocr_data

        return True, analysis, usage, ocr_data

    def _parse_ilr_json(self, ilr_content: Any) -> Dict[str, Any]:
        """Parse ILR into dict, compatible with ```json code block wrapping."""
        if isinstance(ilr_content, dict):
            return ilr_content
        if not isinstance(ilr_content, str):
            raise ValueError(f"ILR type abnormal: {type(ilr_content)}")

        ilr_text = ilr_content.strip()
        if not ilr_text:
            raise ValueError("ILR content is empty")

        if ilr_text.startswith("```"):
            ilr_text = re.sub(r"^```(?:json)?\s*", "", ilr_text, flags=re.IGNORECASE)
            ilr_text = re.sub(r"\s*```$", "", ilr_text)

        try:
            return json.loads(ilr_text)
        except json.JSONDecodeError:
            # Some models attach explanatory text before/after JSON, fallback to extract outermost object
            match = re.search(r"\{.*\}", ilr_text, flags=re.DOTALL)
            if match:
                return json.loads(match.group(0))
            raise

    def _get_first_failed_test(self, test_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get the first failed test case."""
        failed_tests = [r for r in (test_result or {}).get('results', []) if not r.get('passed', False)]
        return failed_tests[0] if failed_tests else None

    def _build_call_expression(self, code: str, input_data: Any, function_name: Optional[str] = None) -> str:
        """Reuse CodeTester's call rules to construct an executable function call expression."""
        detected_class_name, detected_method_name = self.code_tester._extract_solution_info(code)

        if function_name is not None:
            if detected_method_name == function_name:
                class_name = detected_class_name
                method_name = detected_method_name
            else:
                class_name = None
                method_name = function_name
        else:
            class_name = detected_class_name
            method_name = detected_method_name

        if not method_name:
            raise ValueError("Cannot extract function name from code for instrumented execution")

        call_prefix = f"{class_name}().{method_name}" if class_name else method_name

        if isinstance(input_data, dict):
            if 'args' in input_data:
                arg_count = self.code_tester._get_function_arg_count(code, class_name, method_name)
                args_str = json.dumps(input_data['args'], ensure_ascii=False)
                if arg_count == 1:
                    return f"{call_prefix}({args_str})"
                return f"{call_prefix}(*{args_str})"
            if 'kwargs' in input_data:
                kwargs_str = json.dumps(input_data['kwargs'], ensure_ascii=False)
                return f"{call_prefix}(**{kwargs_str})"
            input_dict_str = json.dumps(input_data, ensure_ascii=False)
            return f"{call_prefix}(**{input_dict_str})"

        if isinstance(input_data, str):
            stripped_input = input_data.strip()
            # LCB failures usually report the original call expression.  Do not
            # wrap an existing call in a second invocation (for example,
            # ``solve(solve(...))``); reuse it directly when it targets the
            # selected entry point.
            if stripped_input.startswith(f"{method_name}(") and stripped_input.endswith(")"):
                return stripped_input
            # Original LCB call-based records may contain raw multi-line input.
            # Parse it with the same rules as the LCB tester and invoke the
            # generated function with the resulting arguments.
            if "\n" in stripped_input:
                try:
                    from tools.lcb_tester import LCBCallBasedTester

                    args, kwargs = LCBCallBasedTester()._parse_call_expression(
                        stripped_input, method_name
                    )
                    if kwargs:
                        kwargs_text = ", ".join(
                            f"{key}={repr(value)}" for key, value in kwargs.items()
                        )
                        return f"{call_prefix}({kwargs_text})"
                    return f"{call_prefix}({', '.join(repr(value) for value in args)})"
                except Exception:
                    pass
            try:
                import ast
                parsed = ast.literal_eval(stripped_input)
                if isinstance(parsed, dict):
                    input_dict_str = json.dumps(parsed, ensure_ascii=False)
                    return f"{call_prefix}(**{input_dict_str})"
                return f"{call_prefix}({stripped_input})"
            except Exception:
                return f"{call_prefix}({stripped_input!r})"

        return f"{call_prefix}({repr(input_data)})"

    def _build_instrumented_runner(self, call_expression: str, code_line_count: int) -> str:
        """Construct a test entry with runtime tracing."""
        return f'''
import contextlib
import io
import json
import linecache
import sys
import traceback

TRACE_EVENTS = []
MAX_TRACE_EVENTS = 400
CODE_END_LINE = {code_line_count}
TRACE_START = "__CODEVISION_TRACE_START__"
TRACE_END = "__CODEVISION_TRACE_END__"


def _safe_repr(value, limit=240):
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr_error: {{exc}}>"
    if len(text) > limit:
        return text[:limit] + "...<truncated>"
    return text


def _trace(frame, event, arg):
    if frame.f_code.co_filename != __file__:
        return _trace
    if frame.f_lineno > CODE_END_LINE:
        return _trace
    # Hot loops emit unbounded line events; cap what we record.
    if len(TRACE_EVENTS) >= MAX_TRACE_EVENTS:
        return _trace

    event_info = {{
        "event": event,
        "function": frame.f_code.co_name,
        "line": frame.f_lineno,
        "source": linecache.getline(__file__, frame.f_lineno).rstrip(),
        "locals": {{
            key: _safe_repr(value)
            for key, value in frame.f_locals.items()
            if not key.startswith("__")
        }}
    }}

    if event == "return":
        event_info["return"] = _safe_repr(arg)
    elif event == "exception" and arg:
        exc_type, exc_value, _ = arg
        event_info["exception"] = f"{{getattr(exc_type, '__name__', str(exc_type))}}: {{exc_value}}"

    TRACE_EVENTS.append(event_info)
    return _trace


if __name__ == "__main__":
    payload = {{
        "call_expression": {json.dumps(call_expression, ensure_ascii=False)},
        "captured_stdout": "",
        "captured_stderr": "",
        "result": "",
        "exception_type": "",
        "exception": "",
        "traceback": "",
        "trace": []
    }}
    _stdout_buffer = io.StringIO()
    _stderr_buffer = io.StringIO()

    try:
        sys.settrace(_trace)
        with contextlib.redirect_stdout(_stdout_buffer), contextlib.redirect_stderr(_stderr_buffer):
            _result = {call_expression}
        payload["result"] = _safe_repr(_result, limit=1000)
    except Exception as exc:
        payload["exception_type"] = exc.__class__.__name__
        payload["exception"] = str(exc)
        payload["traceback"] = traceback.format_exc()
    finally:
        sys.settrace(None)
        payload["captured_stdout"] = _stdout_buffer.getvalue()[:4000]
        payload["captured_stderr"] = _stderr_buffer.getvalue()[:4000]
        payload["trace"] = TRACE_EVENTS
        print(TRACE_START)
        print(json.dumps(payload, ensure_ascii=False))
        print(TRACE_END)
'''

    def _build_instrumented_stdio_runner(self, code_line_count: int) -> str:
        """Construct a tracing runner for native stdin/stdout solutions."""
        return f'''
import contextlib
import io
import json
import linecache
import sys
import traceback

TRACE_EVENTS = []
MAX_TRACE_EVENTS = 400
CODE_END_LINE = {code_line_count}
TRACE_START = "__CODEVISION_TRACE_START__"
TRACE_END = "__CODEVISION_TRACE_END__"

def _safe_repr(value, limit=240):
    try:
        text = repr(value)
    except Exception as exc:
        text = f"<repr_error: {{exc}}>"
    return text if len(text) <= limit else text[:limit] + "...<truncated>"

def _trace(frame, event, arg):
    if frame.f_code.co_filename != __file__ or frame.f_lineno > CODE_END_LINE:
        return _trace
    # Hot loops emit unbounded line events; cap what we record.
    if len(TRACE_EVENTS) >= MAX_TRACE_EVENTS:
        return _trace
    item = {{"event": event, "function": frame.f_code.co_name,
            "line": frame.f_lineno,
            "source": linecache.getline(__file__, frame.f_lineno).rstrip(),
            "locals": {{k: _safe_repr(v) for k, v in frame.f_locals.items() if not k.startswith('__')}}}}
    if event == "return":
        item["return"] = _safe_repr(arg)
    elif event == "exception" and arg:
        item["exception"] = f"{{getattr(arg[0], '__name__', str(arg[0]))}}: {{arg[1]}}"
    TRACE_EVENTS.append(item)
    return _trace

if __name__ == "__main__":
    payload = {{"call_expression": "<stdin>", "captured_stdout": "",
                "captured_stderr": "", "result": "", "exception_type": "",
                "exception": "", "traceback": "", "trace": []}}
    out_buffer, err_buffer = io.StringIO(), io.StringIO()
    try:
        sys.settrace(_trace)
        with contextlib.redirect_stdout(out_buffer), contextlib.redirect_stderr(err_buffer):
            exec(compile(open(__file__, encoding="utf-8").read().split("\\nif __name__ == \\\"__main__\\\":", 1)[0], __file__, "exec"), globals())
            _solve = globals().get("solve")
            if callable(_solve):
                _solve()
    except Exception as exc:
        payload["exception_type"] = exc.__class__.__name__
        payload["exception"] = str(exc)
        payload["traceback"] = traceback.format_exc()
    finally:
        sys.settrace(None)
        payload["captured_stdout"] = out_buffer.getvalue()[:4000]
        payload["captured_stderr"] = err_buffer.getvalue()[:4000]
        payload["trace"] = TRACE_EVENTS
        print(TRACE_START)
        print(json.dumps(payload, ensure_ascii=False))
        print(TRACE_END)
'''

    def _run_instrumented_failed_case(self, sample: FailedSample, code: str,
                                      failed_test: Dict[str, Any]) -> Dict[str, Any]:
        """Execute instrumentation on the first failed sample to collect runtime trace."""
        input_data = failed_test.get('input', failed_test.get('input_args', 'N/A'))
        expected_output = failed_test.get('expected_output', failed_test.get('expected', 'N/A'))
        observed_output = failed_test.get('actual_output', failed_test.get('actual', 'N/A'))

        try:
            cleaned_code = self.code_tester._clean_code(code)
            # Native stdio cases must receive the original input through
            # subprocess stdin; only converted solve(...) records use a call.
            if sample.dataset == "LiveCodeBench" and sample.lcb_mode == "stdio":
                raw_input = input_data if isinstance(input_data, str) else str(input_data)
                if not raw_input.strip().startswith("solve("):
                    code_line_count = len(cleaned_code.splitlines())
                    runner_code = self._build_instrumented_stdio_runner(code_line_count)
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as handle:
                        handle.write(cleaned_code)
                        handle.write('\n\n')
                        handle.write(runner_code)
                        temp_file = handle.name
                    try:
                        result = subprocess.run([sys.executable, temp_file], input=raw_input,
                                                capture_output=True, text=True,
                                                encoding='utf-8', timeout=8)
                    finally:
                        try:
                            os.unlink(temp_file)
                        except Exception:
                            pass
                    stdout_text = result.stdout or ''
                    marker_start = stdout_text.find('__CODEVISION_TRACE_START__')
                    marker_end = stdout_text.find('__CODEVISION_TRACE_END__')
                    payload = {}
                    preamble_stdout = stdout_text.strip()
                    if marker_start != -1 and marker_end > marker_start:
                        preamble_stdout = stdout_text[:marker_start].strip()
                        payload_text = stdout_text[marker_start + len('__CODEVISION_TRACE_START__'):marker_end].strip()
                        if payload_text:
                            payload = json.loads(payload_text)
                    return {'input': input_data, 'expected_output': expected_output,
                            'observed_output': observed_output, 'returncode': result.returncode,
                            'module_stdout': preamble_stdout,
                            'subprocess_stderr': result.stderr or '', 'payload': payload}
            call_expression = self._build_call_expression(cleaned_code, input_data, sample.entry_point)
            code_line_count = len(cleaned_code.splitlines())
            runner_code = self._build_instrumented_runner(call_expression, code_line_count)

            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as handle:
                handle.write(cleaned_code)
                handle.write('\n\n')
                handle.write(runner_code)
                temp_file = handle.name

            try:
                result = subprocess.run(
                    [sys.executable, temp_file],
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    timeout=8
                )
            finally:
                try:
                    os.unlink(temp_file)
                except Exception:
                    pass

            stdout_text = result.stdout or ''
            stderr_text = result.stderr or ''
            marker_start = stdout_text.find('__CODEVISION_TRACE_START__')
            marker_end = stdout_text.find('__CODEVISION_TRACE_END__')

            payload = {}
            preamble_stdout = stdout_text.strip()
            if marker_start != -1 and marker_end != -1 and marker_end > marker_start:
                preamble_stdout = stdout_text[:marker_start].strip()
                payload_text = stdout_text[marker_start + len('__CODEVISION_TRACE_START__'):marker_end].strip()
                if payload_text:
                    payload = json.loads(payload_text)

            return {
                'input': input_data,
                'expected_output': expected_output,
                'observed_output': observed_output,
                'returncode': result.returncode,
                'module_stdout': preamble_stdout,
                'subprocess_stderr': stderr_text,
                'payload': payload
            }
        except subprocess.TimeoutExpired:
            return {
                'input': input_data,
                'expected_output': expected_output,
                'observed_output': observed_output,
                'returncode': -1,
                'module_stdout': '',
                'subprocess_stderr': 'Instrumented execution timed out after 8 seconds.',
                'payload': {}
            }
        except Exception as exc:
            return {
                'input': input_data,
                'expected_output': expected_output,
                'observed_output': observed_output,
                'returncode': -1,
                'module_stdout': '',
                'subprocess_stderr': f'Instrumented execution failed: {exc}',
                'payload': {}
            }

    @staticmethod
    def _clip(text: Any, limit: int) -> str:
        """Clip oversized runtime/debug blobs before they enter an LLM prompt."""
        s = text if isinstance(text, str) else str(text)
        if len(s) <= limit:
            return s
        return s[:limit] + f"\n[... clipped: {len(s) - limit} more chars ...]"

    def _format_runtime_trace(self, execution_info: Dict[str, Any]) -> str:
        """Format instrumented execution results as text in the reflection prompt."""
        payload = execution_info.get('payload', {}) or {}
        trace_events = payload.get('trace', []) or []

        # High-value sections first: failing test I/O, exception, then the
        # execution trace with its own budget, then verbose stdout blobs.
        sections = [
            f"Failed Test Input: {self._clip(execution_info.get('input', 'N/A'), 1500)}",
            f"Expected Output: {self._clip(execution_info.get('expected_output', 'N/A'), 1500)}",
            f"Observed Output From Previous Test Run: {self._clip(execution_info.get('observed_output', 'N/A'), 1500)}",
            f"Instrumented Call Expression: {payload.get('call_expression', 'N/A')}",
            f"Subprocess Return Code: {execution_info.get('returncode', 'N/A')}"
        ]
        if payload.get('exception_type') or payload.get('exception'):
            sections.append(
                f"Exception: {payload.get('exception_type', '')} {payload.get('exception', '')}".strip()
            )
        if payload.get('traceback'):
            sections.append(f"Python Traceback:\n{self._clip(payload['traceback'], 2500)}")

        if trace_events:
            # Hot loops emit unbounded line events; a capped prefix already
            # tells the LLM the loop shape, and formatting all events of a
            # runaway loop would bloat the prompt past any upload limit.
            trace_lines = []
            for idx, event in enumerate(trace_events[:150], 1):
                trace_line = (
                    f"{idx}. [{event.get('event', 'unknown')}] "
                    f"{event.get('function', '<unknown>')} @ line {event.get('line', '?')}: "
                    f"{event.get('source', '').strip()}"
                )
                trace_lines.append(trace_line)
                locals_snapshot = event.get('locals', {})
                if locals_snapshot:
                    trace_lines.append(f"   locals: {json.dumps(locals_snapshot, ensure_ascii=False, sort_keys=True)}")
                if event.get('return'):
                    trace_lines.append(f"   return: {event['return']}")
                if event.get('exception'):
                    trace_lines.append(f"   exception: {event['exception']}")
            trace_block = self._clip("\n".join(trace_lines), 10000)
            if len(trace_events) > 150:
                trace_block += f"\n[... {len(trace_events) - 150} more trace events omitted ...]"
            sections.append("Execution Trace:\n" + trace_block)
        else:
            sections.append("Execution Trace: <no trace events captured>")

        if execution_info.get('module_stdout'):
            sections.append(f"Module-level STDOUT before trace payload:\n{self._clip(execution_info['module_stdout'], 2500)}")
        if payload.get('captured_stdout'):
            sections.append(f"Captured STDOUT during function execution:\n{self._clip(payload['captured_stdout'], 2500)}")
        if payload.get('captured_stderr'):
            sections.append(f"Captured STDERR during function execution:\n{self._clip(payload['captured_stderr'], 2500)}")
        if payload.get('result'):
            sections.append(f"Function Return Value: {payload['result']}")
        if execution_info.get('subprocess_stderr'):
            sections.append(f"Subprocess STDERR:\n{self._clip(execution_info['subprocess_stderr'], 1500)}")

        return self._clip("\n\n".join(sections), 22000)

    def _check_code(self, sample: FailedSample) -> Tuple[bool, str, Dict]:
        """Check if the generated code has issues.

        Pass/fail is judged on the full (hidden) test suite; the LLM error
        context is public-test-first (early sample_io reflection design) and
        hidden-test details are withheld while all public cases pass."""
        # If ILR logic is correct but code test fails, it means code generation has issues
        test_result = sample.test_result

        # total==0 means tests could not run; treat as code issue
        if test_result.get('total', 0) == 0 or test_result.get('passed', 0) < test_result.get('total', 0):
            error_summary = self._get_error_context_for_llm(sample)
            if not error_summary:
                error_summary = "Code generation has issues (no failure details available).\n"

            # Collect context info
            context = {
                'error_summary': error_summary
            }

            # The instrumented runtime trace would reveal hidden-test
            # internals; skip it while the public context is withheld.
            if not self._is_public_context_withheld(sample):
                first_failed_test = self._get_first_failed_test(test_result)
                if first_failed_test:
                    execution_info = self._run_instrumented_failed_case(sample, sample.original_code, first_failed_test)
                    context['runtime_trace'] = self._format_runtime_trace(execution_info)

            # Get LLM analysis
            analysis, usage = self._get_llm_analysis(sample, "code", context)
            return True, analysis, usage

        return False, "Code generation is normal", {}

    def _force_code_analysis(self, sample: FailedSample) -> Tuple[str, Dict]:
        """
        When ILR is determined to have no issues, force code analysis (no longer dependent on test failure judgment).

        Error context is public-test-first (early sample_io reflection
        design); hidden-test details are withheld while all public cases pass,
        and the instrumented runtime trace is skipped in that case.

        Returns:
            (analysis, usage)
        """
        error_summary = self._get_error_context_for_llm(sample)
        if not error_summary:
            error_summary = ("ILR seems correct (interpreter verified), but code still "
                             "failed hidden tests. No failure details available.")

        context = {
            'error_summary': error_summary
        }

        if not self._is_public_context_withheld(sample):
            first_failed_test = self._get_first_failed_test(sample.test_result or {})
            if first_failed_test:
                execution_info = self._run_instrumented_failed_case(sample, sample.original_code, first_failed_test)
                context['runtime_trace'] = self._format_runtime_trace(execution_info)

        analysis, usage = self._get_llm_analysis(sample, "code", context)
        return analysis, usage

    def _analyze_code_with_reflection(self,
                                      sample: FailedSample,
                                      ilr: str,
                                      code: str,
                                      test_result: Dict[str, Any]) -> Tuple[str, Dict]:
        """
        Use reflection Agent to analyze current code failure (forced code analysis mode)

        Returns:
            (analysis, usage)
        """
        # When total==0 the test could not run; treat it as a code issue
        if test_result.get('total', 0) > 0 and test_result.get('passed', 0) >= test_result.get('total', 0):
            return "Code generation appears normal", {}

        # Public-test-first error context (early sample_io reflection design)
        error_summary = self._get_error_context_for_llm(sample)
        if not error_summary:
            error_summary = ("Code generation has issues but no failure details "
                             "are available (public tests all passed; hidden "
                             "details withheld).\n")

        context = {
            'error_summary': error_summary
        }

        # The instrumented runtime trace would reveal hidden-test internals;
        # skip it while the public context is withheld.
        if not self._is_public_context_withheld(sample):
            first_failed_test = self._get_first_failed_test(test_result)
            if first_failed_test:
                execution_info = self._run_instrumented_failed_case(sample, code, first_failed_test)
                context['runtime_trace'] = self._format_runtime_trace(execution_info)

        analysis, usage = self._get_llm_analysis(
            sample=sample,
            issue_type="code",
            context=context,
            current_ilr=ilr,
            current_code=code
        )
        return analysis, usage

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Containment similarity: how much of the ILR label's vocabulary is
        covered by an OCR text. The ILR label is a condensed paraphrase of the
        flowchart text, so Jaccard's penalty on extra OCR words made the check
        fire on benign wording differences; containment does not. (The
        controlled evaluation showed 48/50 verified-correct ILRs triggered
        1-12 false mismatches under Jaccard.)"""
        import re
        if not text1 or not text2:
            return 0.0
        words1 = set(re.findall(r'\w+', text1.lower()))
        words2 = set(re.findall(r'\w+', text2.lower()))
        if not words1:
            return 0.0
        return len(words1 & words2) / len(words1)

    def _is_initialization_action(self, action: str) -> bool:
        """Determine if the action is an initialization operation"""
        if not action:
            return False
        # Simple heuristic: actions containing assignment operations may be initializations
        # This can be adjusted based on actual situations
        initialization_patterns = ['=', 'ctx[']
        return any(pattern in action for pattern in initialization_patterns)

    def regenerate_ilr(self, sample: FailedSample, analysis: str, ocr_data: Optional[Dict] = None) -> Tuple[str, Dict]:
        """
        Regenerate ILR

        Args:
            sample: Failed sample
            analysis: Reflection analysis
            ocr_data: Optional OCR data, if provided it will be used directly to avoid re-extraction
        """
        print(f"\nRegenerating ILR with reflection...")
        if not self.ilr_generator:
            return sample.original_ilr, {}

        try:
            if analysis.strip() == "NO_ISSUE":
                return sample.original_ilr, {}
            # Use ILR generator to regenerate, passing original ILR and reflection analysis
            ilr_str, usage, _ = self.ilr_generator.generate_ilr_with_reflection(
                image_path=sample.image_path,
                reflection_prompt=analysis,
                starter_code=sample.starter_code,
                task_id=sample.task_id,
                dataset=sample.dataset,
                ocr_data=ocr_data,  # Pass OCR data
                original_ilr=sample.original_ilr,  # Pass original ILR
                problem_text_description=sample.problem_text_description,
            )

            return ilr_str, usage

        except Exception as e:
            print(f"ILR regeneration failed: {e}")
            return sample.original_ilr, {}

    def regenerate_code_normal(self, sample: FailedSample, ilr: str) -> Tuple[str, Dict]:
        """Generate code normally from new ILR (without reflection)"""
        print(f"\nRegenerating code from new ILR (normal mode)...")

        try:
            problem_info = {
                'entry_point': sample.entry_point,
                'test_cases': sample.test_cases,
                'starter_code': sample.starter_code,
                'text_description': sample.problem_text_description,
            }

            code_str, usage = self.code_generator.generate_code(
                ilr=ilr,
                problem_info=problem_info,
                representation_type=self.intermediate_representation_type,
            )

            return code_str, usage

        except Exception as e:
            print(f"Normal code generation failed: {e}")
            return sample.original_code, {}

    def regenerate_code(self, sample: FailedSample, ilr: str, analysis: str, current_iteration: int = 1) -> Tuple[str, Dict]:
        """Regenerate code (with reflection)"""
        print(f"\nRegenerating code with reflection...")

        try:
            # Build problem_info dict
            problem_info = {
                'entry_point': sample.entry_point,
                'test_cases': sample.test_cases,
                'starter_code': sample.starter_code,
                'text_description': sample.problem_text_description,
            }

            if self.reflection_mode == "code_only":
                # In code_only mode, only pass image in the first iteration
                image_path = sample.image_path if current_iteration == 1 else None
                code_str, usage = self.code_generator.generate_code_from_code_reflection(
                    problem_info=problem_info,
                    original_code=sample.original_code,
                    analysis=analysis,
                    image_path=image_path
                )
            else:
                # Use code generator to regenerate, passing original code and reflection analysis
                code_str, usage = self.code_generator.generate_code_with_reflection(
                    ilr=ilr,
                    problem_info=problem_info,
                    original_code=sample.original_code,
                    analysis=analysis,
                    representation_type=self.intermediate_representation_type,
                )

            return code_str, usage

        except Exception as e:
            print(f"Code regeneration failed: {e}")
            return sample.original_code, {}

    def test_single_sample(self, sample: FailedSample, code: str) -> Dict:
        """Test a single sample."""
        print("\nTesting generated code...")
        if sample.dataset == "LiveCodeBench":
            return self._test_lcb_sample(sample, code)
        return self.code_tester.test_single_problem(
            generated_code=code,
            problem_info=self._build_problem_info_for_testing(sample),
        )
    def process_sample(self, sample: FailedSample) -> List[IterationResult]:
        """
        Process a single failed sample, supporting multiple iterations

        Optimization strategy:
        - 1st iteration: Normal analysis (OCR+ILR check -> Code check)
        - If 1st analysis result is ILR issue: 2nd+ iterations always do code reflection analysis (no longer fall back to ILR analysis)
        - When issue_type is unknown, terminate early
        - ilr_only mode: each iteration only analyzes ILR, after regenerating ILR use normal code generation

        Returns:
            List of all iteration results
        """
        print(f"\n{'#'*60}")
        print(f"Processing sample: {sample.task_id}")
        print(f"{'#'*60}")

        iteration_results = []
        current_ilr = sample.original_ilr
        current_code = sample.original_code
        current_test_result = sample.test_result
        force_code_reflection = self.reflection_mode == "code_only"

        # === Pre-check (early sample_io reflection design): run the current
        # code against the PUBLIC test cases to collect the LLM error
        # context. This never affects pass/fail judgment, which always uses
        # the full (hidden) test suite. ===
        if sample.sample_io:
            print("\n[Pre-check] Testing the original code against public test cases (sample_io)...")
            sample.sample_io_result = self._test_with_sample_io(
                current_code, sample.sample_io, sample.entry_point)
            r = sample.sample_io_result
            print(f"  sample_io result: {r.get('passed', 0)}/{r.get('total', 0)} passed")
            if r.get('total', 0) > 0 and r.get('passed', 0) == r.get('total', 0):
                print("  All public cases pass; hidden-test details will be withheld from the LLM")
            else:
                failed_public = [x for x in r.get('results', []) if not x.get('passed', False)]
                if failed_public:
                    print(f"  First failing public case: {failed_public[0].get('assertion', '')}")
                    print(f"  Error: {failed_public[0].get('error', '')}")
        else:
            print("\n[Pre-check] No sample_io data for this task; falling back to full-test error context")
            sample.sample_io_result = None

        for iteration in range(1, self.max_iterations + 1):
            iter_start_time = time.time()
            total_tokens = {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}

            print(f"\n{'='*60}")
            print(f"Iteration {iteration}/{self.max_iterations}")
            print(f"{'='*60}")

            # Before each analysis round, sync current state to ensure reflection sees the latest ILR/Code/Test from the previous round
            sample.original_ilr = current_ilr
            sample.original_code = current_code
            sample.test_result = current_test_result
            # Refresh the public-test context for the current code (analysis
            # only; pass/fail judgment is unaffected)
            if sample.sample_io:
                sample.sample_io_result = self._test_with_sample_io(
                    current_code, sample.sample_io, sample.entry_point)

            if force_code_reflection and (iteration > 1 or self.reflection_mode == "code_only"):
                # After first iteration determines ILR issue, subsequent iterations no longer fall back to ILR analysis, always do code reflection analysis
                print("Current iteration is fixed to code reflection analysis...")
                issue_type = "code"
                if self.reflection_mode == "code_only":
                    analysis, analysis_usage = self._analyze_code_only_ablation(
                        sample=sample,
                        code=current_code,
                        test_result=current_test_result
                    )
                else:
                    analysis, analysis_usage = self._analyze_code_with_reflection(
                        sample=sample,
                        ilr=current_ilr,
                        code=current_code,
                        test_result=current_test_result
                    )
                ocr_data = None
            else:
                # Normal analysis
                issue_type, analysis, analysis_usage, ocr_data = self.analyze_sample(sample)

            # After first iteration analysis finds ILR issue, subsequent iterations always do code reflection analysis (unless in ilr_only mode)
            if iteration == 1 and issue_type == "ilr" and self.reflection_mode != "ilr_only":
                force_code_reflection = True

            # Accumulate analysis phase token consumption
            if analysis_usage:
                total_tokens['prompt_tokens'] += analysis_usage.get('prompt_tokens', 0)
                total_tokens['completion_tokens'] += analysis_usage.get('completion_tokens', 0)
                total_tokens['total_tokens'] += analysis_usage.get('total_tokens', 0)

            print(f"\nIssue type: {issue_type}")
            print(f"Analysis result:\n{analysis}")

            # unknown type: cannot determine issue, terminate early to avoid wasting iterations
            if issue_type == "unknown":
                print(f"\nUnable to determine issue type, terminating iteration")
                iter_end_time = time.time()
                raw_duration = iter_end_time - iter_start_time
                # Deduct analysis phase wait time
                wait = analysis_usage.get('total_wait_time', 0) if isinstance(analysis_usage, dict) else 0
                duration = max(0.0, raw_duration - wait)
                iter_result = IterationResult(
                    iteration=iteration,
                    task_id=sample.task_id,
                    issue_type=issue_type,
                    analysis=analysis,
                    regenerated_ilr=None,
                    regenerated_code=None,
                    test_result=current_test_result,
                    passed=False,
                    token_usage=total_tokens,
                    duration=duration
                )
                iteration_results.append(iter_result)
                break

            # Process based on issue type
            regenerated_ilr = None
            regenerated_code = None

            if issue_type == "ilr":
                # Regenerate ILR, pass OCR data to avoid re-extraction
                regenerated_ilr, ilr_usage = self.regenerate_ilr(sample, analysis, ocr_data)
                current_ilr = regenerated_ilr

                # Accumulate tokens (ilr_usage may be a nested dict)
                if ilr_usage:
                    # Check if it contains usage for sub-steps
                    if 'ocr_extraction' in ilr_usage or 'ilr_generation' in ilr_usage:
                        # This is a nested dict
                        for key, usage in ilr_usage.items():
                            if isinstance(usage, dict):
                                total_tokens['prompt_tokens'] += usage.get('prompt_tokens', 0)
                                total_tokens['completion_tokens'] += usage.get('completion_tokens', 0)
                                total_tokens['total_tokens'] += usage.get('total_tokens', 0)
                    else:
                        # This is a flat dict (backward compatible)
                        total_tokens['prompt_tokens'] += ilr_usage.get('prompt_tokens', 0)
                        total_tokens['completion_tokens'] += ilr_usage.get('completion_tokens', 0)
                        total_tokens['total_tokens'] += ilr_usage.get('total_tokens', 0)

                # ILR has been regenerated, use normal code generator (without reflection)
                regenerated_code, code_usage = self.regenerate_code_normal(sample, current_ilr)
                current_code = regenerated_code

                # Accumulate tokens
                if code_usage:
                    total_tokens['prompt_tokens'] += code_usage.get('prompt_tokens', 0)
                    total_tokens['completion_tokens'] += code_usage.get('completion_tokens', 0)
                    total_tokens['total_tokens'] += code_usage.get('total_tokens', 0)

            elif issue_type == "code":
                # Only regenerate code
                regenerated_code, code_usage = self.regenerate_code(sample, current_ilr, analysis, iteration)
                current_code = regenerated_code

                # Accumulate tokens
                if code_usage:
                    total_tokens['prompt_tokens'] += code_usage.get('prompt_tokens', 0)
                    total_tokens['completion_tokens'] += code_usage.get('completion_tokens', 0)
                    total_tokens['total_tokens'] += code_usage.get('total_tokens', 0)

            # Test new code
            test_result = self.test_single_sample(sample, current_code)
            current_test_result = test_result

            # Only consider passed when there are test cases and all pass
            _passed_cnt = test_result.get('passed', 0)
            _total_cnt = test_result.get('total', 0)
            passed = (_total_cnt > 0) and (_passed_cnt == _total_cnt)

            iter_end_time = time.time()
            raw_duration = iter_end_time - iter_start_time

            # Deduct API wait time, only count actual processing time
            total_wait = 0
            for usage_dict in [analysis_usage]:
                if isinstance(usage_dict, dict):
                    total_wait += usage_dict.get('total_wait_time', 0)
            # Also deduct generation phase wait time
            if issue_type == "ilr":
                if ilr_usage:
                    if 'ocr_extraction' in ilr_usage or 'ilr_generation' in ilr_usage:
                        for key, u in ilr_usage.items():
                            if isinstance(u, dict):
                                total_wait += u.get('total_wait_time', 0)
                    else:
                        total_wait += ilr_usage.get('total_wait_time', 0)
                if code_usage:
                    total_wait += code_usage.get('total_wait_time', 0)
            elif issue_type == "code":
                if code_usage:
                    total_wait += code_usage.get('total_wait_time', 0)
            duration = max(0.0, raw_duration - total_wait)

            print(f"\nTest result: {'Passed' if passed else 'Failed'}")
            print(f"Pass rate: {test_result.get('pass_rate', 0):.1f}%")
            print(f"Iteration duration: {duration:.2f}s")
            print(f"Token consumption: {total_tokens}")

            # Record iteration result
            iter_result = IterationResult(
                iteration=iteration,
                task_id=sample.task_id,
                issue_type=issue_type,
                analysis=analysis,
                regenerated_ilr=regenerated_ilr,
                regenerated_code=regenerated_code,
                test_result=test_result,
                passed=passed,
                token_usage=total_tokens,
                duration=duration
            )

            iteration_results.append(iter_result)

            # If passed, stop iteration
            if passed:
                print(f"\n✓ Sample {sample.task_id} passed after iteration {iteration}!")
                break

            # Update sample current state
            sample.original_ilr = current_ilr
            sample.original_code = current_code
            sample.test_result = current_test_result

        return iteration_results

    def save_single_iteration_result(self, iteration_result: IterationResult):
        """Save a single iteration result (append mode)"""
        iteration = iteration_result.iteration

        # 1. Save to samples{iteration}.jsonl
        samples_file = self.output_dir / f"samples{iteration}.jsonl"
        samples_file.parent.mkdir(parents=True, exist_ok=True)

        sample_result = {
            'task_id': iteration_result.task_id,
            'iteration': iteration,
            'issue_type': iteration_result.issue_type,
            'analysis': iteration_result.analysis,
            'generated_code': iteration_result.regenerated_code,
            'test_result': iteration_result.test_result,
            'passed': iteration_result.passed,
            'token_usage': iteration_result.token_usage,
            'duration': iteration_result.duration
        }

        with open(samples_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(sample_result, ensure_ascii=False) + '\n')

        # 2. Save to ilr{iteration}.jsonl
        if iteration_result.regenerated_ilr:
            ilr_file = self.output_dir / f"ilr{iteration}.jsonl"
            ilr_result = {
                iteration_result.task_id: iteration_result.regenerated_ilr
            }
            with open(ilr_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(ilr_result, ensure_ascii=False) + '\n')

        # 3. Save to samples.jsonl_results{iteration}.jsonl
        test_results_file = self.output_dir / f"samples.jsonl_results{iteration}.jsonl"
        test_result_entry = {
            'task_id': iteration_result.task_id,
            'passed': iteration_result.passed,
            'test_result': iteration_result.test_result or {}
        }
        with open(test_results_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(test_result_entry, ensure_ascii=False) + '\n')

    def check_if_processed(self, task_id: str, max_iterations: int) -> bool:
        """Check if the task has already been processed"""
        # Check if the last iteration file has a record for this task
        # If any iteration passed, or max iterations were reached, it is considered processed

        # First check if there is a passed record
        for i in range(1, max_iterations + 1):
            results_file = self.output_dir / f"samples.jsonl_results{i}.jsonl"
            if results_file.exists():
                try:
                    with open(results_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if not line.strip(): continue
                            data = json.loads(line)
                            if data.get('task_id') == task_id:
                                if data.get('passed', False):
                                    return True  # Successfully processed
                except:
                    pass

        # If no successful record, check if max iterations have been completed
        # Only need to check if the last iteration file has a record
        last_iter_file = self.output_dir / f"samples.jsonl_results{max_iterations}.jsonl"
        if last_iter_file.exists():
            try:
                with open(last_iter_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        if not line.strip(): continue
                        data = json.loads(line)
                        if data.get('task_id') == task_id:
                            return True  # All iterations have been attempted
            except:
                pass

        return False

    def _run_final_merge_and_evaluate(self, n_workers: int = 4, timeout: float = 3.0) -> Optional[Dict[str, Any]]:
        """
        After batch reflection ends, backfill iteration code into original samples.jsonl and perform final evaluation.
        LiveCodeBench uses its mixed I/O evaluator; other datasets use HumanEval evaluation.
        Will generate:
        - samples_reflection_merged.jsonl
        - samples_reflection_merged.jsonl_results.jsonl
        - results.txt
        - reflection_merge_eval_summary.json
        """
        from utils.merge_reflection_and_evaluate import (
            read_jsonl,
            write_jsonl,
            infer_problem_file,
            build_iteration_maps,
            merge_samples,
            run_humaneval,
            build_task_difficulty_map,
            compute_difficulty_stats,
            format_results_txt,
        )

        has_iteration_files = any(
            (self.output_dir / f"samples{i}.jsonl").exists()
            or (self.output_dir / f"samples.jsonl_results{i}.jsonl").exists()
            for i in range(1, self.max_iterations + 1)
        )
        if not has_iteration_files:
            print("No reflection iteration files detected, skipping final merge+evaluate.")
            return None

        samples_file = self.output_dir / "samples.jsonl"
        if not samples_file.exists():
            # results_file is usually samples.jsonl_results.jsonl, here we auto-infer samples.jsonl
            guessed_samples = Path(str(self.results_file).replace("_results.jsonl", ""))
            if guessed_samples.exists():
                samples_file = guessed_samples
            else:
                print(f"Original samples file not found, skipping final merge+evaluate: {samples_file}")
                return None

        data_root = self.data_dir.parent
        dataset_name = self.data_dir.name
        if dataset_name == "LiveCodeBench":
            # LCB evaluation needs func_name/public_test_cases from the merged file;
            # do not use the HumanEval-style inferred problem file.
            problem_file = self.data_dir / "LiveCodeBench_merged.jsonl"
        else:
            problem_file = infer_problem_file(samples_file=samples_file, data_root=data_root)
        if not problem_file.exists():
            # Prefer trying standard naming under data_dir
            dataset_problem = (
                self.data_dir / "HumanEval.jsonl"
                if dataset_name == "HumanEval-V"
                else self.data_dir / f"{dataset_name}.jsonl"
            )
            if dataset_problem.exists():
                problem_file = dataset_problem
            else:
                print(f"Problem file not found, skipping final merge+evaluate: {problem_file}")
                return None

        output_file = self.output_dir / "samples_reflection_merged.jsonl"

        print("\n[Final] Starting merge reflection results and evaluation ...")
        original_rows = read_jsonl(samples_file)
        sample_maps, result_maps = build_iteration_maps(
            reflection_dir=self.output_dir,
            max_iterations=self.max_iterations,
        )
        merged_rows, merge_summary = merge_samples(
            original_rows=original_rows,
            sample_maps=sample_maps,
            result_maps=result_maps,
            max_iterations=self.max_iterations,
        )
        write_jsonl(output_file, merged_rows)

        if dataset_name == "LiveCodeBench":
            # Evaluate merged LCB samples with the mixed I/O evaluator (stdio + call_based).
            from evaluation.evaluate_lcb import evaluate_lcb_samples

            pass_at_1, lcb_results = evaluate_lcb_samples(
                str(output_file), str(problem_file), timeout=max(timeout, 6.0)
            )
            merged_results_file = Path(str(output_file) + "_results.jsonl")
            with merged_results_file.open("w", encoding="utf-8") as handle:
                for row in lcb_results:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            results_txt_path = self.output_dir / "results.txt"
            with results_txt_path.open("w", encoding="utf-8") as handle:
                handle.write(f"{{'pass@1': {pass_at_1}}}\n")
            summary_path = self.output_dir / "reflection_merge_eval_summary.json"
            summary = {
                "samples_file": str(samples_file),
                "reflection_dir": str(self.output_dir),
                "output_file": str(output_file),
                "merged_results_file": str(merged_results_file),
                "results_txt": str(results_txt_path),
                "problem_file": str(problem_file),
                "merge_summary": merge_summary,
                "eval_result": {"pass@1": pass_at_1},
            }
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, ensure_ascii=False, indent=2)
            print("[Final] LCB merge+evaluate completed")
            print(f"  pass@1: {pass_at_1:.6f}")
            print(f"  results_txt: {results_txt_path}")
            return summary

        eval_result = run_humaneval(
            sample_file=str(output_file),
            k=[1],
            n_workers=n_workers,
            timeout=timeout,
            problem_file=str(problem_file),
        )

        merged_results_file = Path(str(output_file) + "_results.jsonl")
        results_rows = read_jsonl(merged_results_file)
        task_to_difficulty = build_task_difficulty_map(problem_file)
        difficulty_stats = compute_difficulty_stats(results_rows, task_to_difficulty)

        results_txt_path = self.output_dir / "results.txt"
        with results_txt_path.open("w", encoding="utf-8") as f:
            f.write(format_results_txt(eval_result, difficulty_stats))

        summary_path = self.output_dir / "reflection_merge_eval_summary.json"
        summary = {
            "samples_file": str(samples_file),
            "reflection_dir": str(self.output_dir),
            "output_file": str(output_file),
            "merged_results_file": str(merged_results_file),
            "results_txt": str(results_txt_path),
            "problem_file": str(problem_file),
            "merge_summary": merge_summary,
            "eval_result": eval_result,
            "difficulty_summary": difficulty_stats,
        }
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print("[Final] merge+evaluate completed")
        print(f"  pass@1: {float(eval_result.get('pass@1', 0.0)):.6f}")
        for level in ("Easy", "Medium", "Hard"):
            passed = int(difficulty_stats[level]["passed"])
            total = int(difficulty_stats[level]["total"])
            rate = float(difficulty_stats[level]["rate"])
            print(f"  {level}: {passed}/{total} = {rate}")
        print(f"  results_txt: {results_txt_path}")

        return summary

    def batch_process(
        self,
        limit: int = None,
        run_final_merge_eval: bool = True,
        eval_n_workers: int = 4,
        eval_timeout: float = 3.0,
    ) -> Dict:
        """
        Batch process all failed samples

        Args:
            limit: Limit the number of samples to process
            run_final_merge_eval: Whether to automatically perform merge + final evaluation after completion
            eval_n_workers: Final evaluation concurrency
            eval_timeout: Final evaluation per-sample timeout (seconds)

        Returns:
            Processing result statistics
        """
        # Get failed samples
        failed_samples = self.get_failed_samples()

        # Difficulty map from the FULL failed set (before the limit slice and
        # the pending filter) — results_iteration.txt is rebuilt from the
        # on-disk iteration files, so its difficulty breakdown must cover
        # tasks processed by earlier runs, not just the subset this
        # invocation happens to pick up.
        all_task_difficulty = {s.task_id: s.difficulty for s in failed_samples}

        if limit:
            failed_samples = failed_samples[:limit]

        total = len(failed_samples)
        print(f"\nFound {total} failed samples")

        if total == 0:
            summary = {
                'total': 0,
                'processed': 0,
                'success': 0,
                'failed': 0
            }
            if run_final_merge_eval:
                final_summary = self._run_final_merge_and_evaluate(
                    n_workers=eval_n_workers,
                    timeout=eval_timeout,
                )
                if final_summary:
                    summary["final_merge_eval"] = final_summary
            return summary

        # Filter already processed samples
        pending_samples = []
        for sample in failed_samples:
            if not self.check_if_processed(sample.task_id, self.max_iterations):
                pending_samples.append(sample)
            else:
                print(f"Skipping already processed task: {sample.task_id}")

        print(f"Remaining tasks to process: {len(pending_samples)}")
        failed_samples = pending_samples

        success_count = 0
        failed_count = 0

        # Statistics variables
        iteration_stats = {i: {'tokens': [], 'duration': []} for i in range(1, self.max_iterations + 1)}

        # Process each sample
        for idx, sample in enumerate(failed_samples, 1):
            print(f"\n{'#'*60}")
            print(f"Processing progress: {idx}/{len(failed_samples)}")
            print(f"{'#'*60}")

            try:
                # Process sample
                iteration_results = self.process_sample(sample)

                # Save and organize results in real-time
                for iter_result in iteration_results:
                    # Save single result in real-time
                    self.save_single_iteration_result(iter_result)

                    iteration = iter_result.iteration

                    # Collect statistics
                    if iter_result.token_usage:
                        iteration_stats[iteration]['tokens'].append(iter_result.token_usage)
                    iteration_stats[iteration]['duration'].append(iter_result.duration)

                # Statistics
                if iteration_results and iteration_results[-1].passed:
                    success_count += 1
                else:
                    failed_count += 1

            except Exception as e:
                print(f"Failed to process sample {sample.task_id}: {e}")
                failed_count += 1

        # Calculate average statistics
        avg_stats = {}
        for iteration in range(1, self.max_iterations + 1):
            stats = iteration_stats[iteration]
            count = len(stats['duration'])

            if count > 0:
                avg_duration = sum(stats['duration']) / count

                # Token statistics
                total_tokens_sum = sum(t.get('total_tokens', 0) for t in stats['tokens'])
                prompt_tokens_sum = sum(t.get('prompt_tokens', 0) for t in stats['tokens'])
                completion_tokens_sum = sum(t.get('completion_tokens', 0) for t in stats['tokens'])

                avg_stats[iteration] = {
                    'count': count,
                    'avg_duration': avg_duration,
                    'avg_total_tokens': total_tokens_sum / count,
                    'avg_prompt_tokens': prompt_tokens_sum / count,
                    'avg_completion_tokens': completion_tokens_sum / count
                }

        # Calculate average duration per iteration (0.0 even if no data)
        avg_duration_by_iteration = {}
        for iteration in range(1, self.max_iterations + 1):
            durations = iteration_stats[iteration]['duration']
            avg_duration_by_iteration[iteration] = sum(durations) / len(durations) if durations else 0.0

        # Note: No longer need batch save here since real-time save is already done
        # But keep the final report generation logic for compatibility

        # Return statistics
        summary = {
            'total': total, # This is the original total count
            'processed': len(failed_samples), # This is the count processed this time
            'success': success_count,
            'failed': failed_count,
            'success_rate': success_count / len(failed_samples) * 100 if failed_samples else 0,
            'iteration_stats': avg_stats,
            'avg_duration_by_iteration': avg_duration_by_iteration,
            'reflection_mode': self.reflection_mode,
        }

        print(f"\n{'='*60}")
        print(f"Batch processing completed!")
        print(f"{'='*60}")
        print(f"Processed this run: {len(failed_samples)}")
        print(f"Success: {success_count}")
        print(f"Failed: {failed_count}")

        # Generate results_iteration.txt from the on-disk iteration files
        # (samples{N}.jsonl + samples.jsonl_results{N}.jsonl) so the report
        # always reflects every run so far, including resumed tasks — not
        # just the samples processed by this invocation.
        self._write_summary_report(all_task_difficulty)

        if run_final_merge_eval:
            final_summary = self._run_final_merge_and_evaluate(
                n_workers=eval_n_workers,
                timeout=eval_timeout,
            )
            if final_summary:
                summary["final_merge_eval"] = final_summary

        return summary

    def _write_summary_report(self, task_difficulty_map=None):
        """Generate detailed statistics report.

        The report is rebuilt from the on-disk iteration files
        (samples{N}.jsonl + samples.jsonl_results{N}.jsonl) so it covers every
        task reflection has ever processed across runs (resume included), not
        just the samples handled by the current invocation.

        Field sources:
        - pass status: samples.jsonl_results{N}.jsonl (task_id/passed), falling
          back to samples{N}.jsonl for tasks missing there (a crash between the
          two writes can leave samples{N} with one extra row);
        - tokens/duration: samples{N}.jsonl (token_usage/duration);
        - difficulty: task_difficulty_map built from the full failed set.

        When a task has several rows in the same iteration file (a re-run after
        a crash re-appends), the LAST row is the latest attempt and wins.
        """
        if task_difficulty_map is None:
            task_difficulty_map = {}

        def _read_rows(path):
            rows = []
            if path.exists():
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                rows.append(json.loads(line))
                            except json.JSONDecodeError:
                                continue
                except OSError:
                    pass
            return rows

        # task -> last row of samples{N}.jsonl (latest attempt)
        last_sample_rows = {}
        # task -> passed flag from the last row of samples.jsonl_results{N}.jsonl
        last_result_pass = {}
        all_tasks = set()

        for i in range(1, self.max_iterations + 1):
            sample_map = {}
            for row in _read_rows(self.output_dir / f"samples{i}.jsonl"):
                task_id = row.get('task_id')
                if task_id:
                    sample_map[task_id] = row  # later rows overwrite earlier ones
            result_map = {}
            for row in _read_rows(self.output_dir / f"samples.jsonl_results{i}.jsonl"):
                task_id = row.get('task_id')
                if task_id:
                    result_map[task_id] = row

            last_sample_rows[i] = sample_map
            last_result_pass[i] = {
                task_id: bool(row.get('passed', False))
                for task_id, row in result_map.items()
            }
            all_tasks.update(sample_map.keys())
            all_tasks.update(result_map.keys())

        all_tasks = sorted(all_tasks)
        total = len(all_tasks)

        # Count totals per difficulty over every touched task
        difficulty_counts = {}
        for task_id in all_tasks:
            diff = task_difficulty_map.get(task_id, 'Unknown')
            difficulty_counts[diff] = difficulty_counts.get(diff, 0) + 1

        # Determine at which iteration each task first passed (latest attempt
        # per iteration; the results file takes precedence over samples{N}).
        passed_at_iteration = {}
        for task_id in all_tasks:
            for i in range(1, self.max_iterations + 1):
                if task_id in last_result_pass[i]:
                    passed = last_result_pass[i][task_id]
                elif task_id in last_sample_rows[i]:
                    passed = bool(last_sample_rows[i][task_id].get('passed', False))
                else:
                    continue
                if passed:
                    passed_at_iteration[task_id] = i
                    break

        overall_success = len(passed_at_iteration)

        report_file = self.output_dir / "results_iteration.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"Batch Reflection Summary Report\n")
            f.write(f"{'='*50}\n\n")

            if total == 0:
                f.write("No iteration results found on disk.\n")
            else:
                f.write(f"Total Samples Processed: {total}\n")
                f.write(f"Overall Success Rate: {overall_success}/{total} ({overall_success/total*100:.2f}%)\n\n")

            # Pass rate improvement statistics
            # Calculate cumulative pass count per iteration and per difficulty
            cumulative_success = 0
            cumulative_success_by_diff = {d: 0 for d in difficulty_counts}

            for i in range(1, self.max_iterations + 1):
                f.write(f"Iteration {i}:\n")
                f.write(f"{'-'*20}\n")

                # 1. Average tokens and time (latest attempt per task)
                durations = []
                tokens = []
                for row in last_sample_rows[i].values():
                    duration = row.get('duration')
                    if isinstance(duration, (int, float)):
                        durations.append(duration)
                    usage = row.get('token_usage')
                    if isinstance(usage, dict):
                        tokens.append(usage)
                count = len(durations)

                if count > 0:
                    f.write(f"Average Duration: {sum(durations) / count:.2f}s\n")
                    f.write(f"Average Tokens:\n")
                    if tokens:
                        total_sum = sum(t.get('total_tokens', 0) for t in tokens)
                        prompt_sum = sum(t.get('prompt_tokens', 0) for t in tokens)
                        completion_sum = sum(t.get('completion_tokens', 0) for t in tokens)
                        f.write(f"  - Total: {total_sum / len(tokens):.1f}\n")
                        f.write(f"  - Prompt: {prompt_sum / len(tokens):.1f}\n")
                        f.write(f"  - Completion: {completion_sum / len(tokens):.1f}\n")
                        f.write(f"Total Tokens:\n")
                        f.write(f"  - Total: {total_sum}\n")
                        f.write(f"  - Prompt: {prompt_sum}\n")
                        f.write(f"  - Completion: {completion_sum}\n")
                    else:
                        f.write(f"  - Total: 0.0\n")
                        f.write(f"  - Prompt: 0.0\n")
                        f.write(f"  - Completion: 0.0\n")
                        f.write(f"Total Tokens:\n")
                        f.write(f"  - Total: 0\n")
                        f.write(f"  - Prompt: 0\n")
                        f.write(f"  - Completion: 0\n")
                else:
                    f.write("No data available for this iteration.\n")

                # 2. Pass rate statistics
                # Find samples that passed in this iteration
                current_iter_success = 0
                current_iter_success_by_diff = {d: 0 for d in difficulty_counts}

                for task_id, passed_iter in passed_at_iteration.items():
                    if passed_iter == i:
                        current_iter_success += 1
                        diff = task_difficulty_map.get(task_id, 'Unknown')
                        if diff in current_iter_success_by_diff:
                            current_iter_success_by_diff[diff] += 1

                cumulative_success += current_iter_success
                for d in current_iter_success_by_diff:
                    cumulative_success_by_diff[d] += current_iter_success_by_diff[d]

                # Calculate improvement rate (percentage relative to total)
                improvement = current_iter_success / total * 100 if total > 0 else 0
                cumulative_rate = cumulative_success / total * 100 if total > 0 else 0

                f.write(f"\nPass Rate Statistics:\n")
                f.write(f"  New Passed: {current_iter_success} (+{improvement:.2f}%)\n")
                f.write(f"  Cumulative Passed: {cumulative_success}/{total} ({cumulative_rate:.2f}%)\n")

                # Statistics by difficulty
                f.write(f"\n  Breakdown by Difficulty:\n")
                sorted_diffs = sorted(difficulty_counts.keys()) # Simple sort, Easy may not always be first, but sufficient

                for diff in sorted_diffs:
                    total_diff = difficulty_counts[diff]
                    new_pass = current_iter_success_by_diff.get(diff, 0)
                    cum_pass = cumulative_success_by_diff.get(diff, 0)

                    diff_improvement = new_pass / total_diff * 100 if total_diff > 0 else 0
                    diff_cum_rate = cum_pass / total_diff * 100 if total_diff > 0 else 0

                    f.write(f"    - {diff}: +{new_pass} ({diff_improvement:.2f}%) -> Total: {cum_pass}/{total_diff} ({diff_cum_rate:.2f}%)\n")

                f.write("\n")

        print(f"Detailed statistics report saved to: {report_file}")



def main():
    """Main function"""
    import argparse

    # Project path constants
    _FILE = Path(__file__).resolve()
    _SRC_DIR = _FILE.parent.parent           # src/
    _PROJECT_ROOT = _SRC_DIR.parent           # flowchart2code/
    _CONFIGS_DIR = _SRC_DIR / "configs"      # src/configs/
    _DATA_DIR = _PROJECT_ROOT / "data"        # data/
    _OUTPUT_DIR = _PROJECT_ROOT / "output"    # output/

    parser = argparse.ArgumentParser(description='Batch Reflection Agent - Process failed test samples')
    parser.add_argument('--config',
                       default=str(_CONFIGS_DIR / "qwen_api_key_config.json"),
                       help='Config file path')
    parser.add_argument('--results',
                       default=str(_OUTPUT_DIR / "HumanEval-V" / "qwen3-vl-plus-qwen3-vl-plus" / "samples.jsonl_results.jsonl"),
                       help='Results file path')
    parser.add_argument('--data-dir',
                       default=str(_DATA_DIR / "HumanEval-V"),
                       help='Data directory')
    parser.add_argument('--output-dir',
                       default=None,
                       help='Output directory (defaults to same directory as results)')
    parser.add_argument('--limit', type=int, default=None,
                       help='Limit the number of samples to process')
    parser.add_argument('--max-iterations', type=int, default=3,
                       help='Maximum number of iterations')
    parser.add_argument('--reflection-mode',
                       choices=['full', 'code_only', 'ilr_only'],
                       default='full',
                       help='Reflection mode: full=normal batch reflection, code_only=code reflection ablation (no ILR generation), ilr_only=ILR reflection ablation (only reflect on ILR)')
    parser.add_argument('--intermediate-representation',
                       choices=['ilr', 'text'],
                       default='ilr',
                       help='Intermediate representation type: ilr or text')
    parser.add_argument('--prompt-variant',
                       choices=['default', 'self_planning'],
                       default='default',
                       help='Prompt variant: default or self_planning')
    parser.add_argument('--text-ir-dir',
                       default=None,
                       help='Text intermediate representation directory (containing text_ir.jsonl), can be used to load Text IR during reflection')
    parser.add_argument('--skip-final-eval', action='store_true',
                       help='Skip the final merge + evaluate')
    parser.add_argument('--eval-workers', type=int, default=4,
                       help='Final evaluation concurrent worker count (default 4)')
    parser.add_argument('--eval-timeout', type=float, default=3.0,
                       help='Final evaluation per-sample timeout in seconds (default 3.0)')
    parser.add_argument('--include-problem-text-description',
                       type=str2bool,
                       nargs='?',
                       const=True,
                       default=True,
                       help='Whether to include the problem text description from the dataset as additional context for ILR/reflection process, default true; can pass false to disable')

    args = parser.parse_args()

    # Force UTF-8 on stdout/stderr to avoid GBK codec errors on Windows
    # when the underlying console (e.g. PowerShell with code page 936)
    # cannot encode characters like ✓ (U+2713) printed by the agent.
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except (AttributeError, ValueError):
        # Python < 3.7 or already detached streams: fall back to wrapping
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

    # Create Agent
    agent = BatchReflectionAgent(
        config_path=args.config,
        results_file=args.results,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        max_iterations=args.max_iterations,
        reflection_mode=args.reflection_mode,
        intermediate_representation_type=args.intermediate_representation,
        text_ir_dir=args.text_ir_dir,
        prompt_variant=args.prompt_variant,
        include_problem_text_description=args.include_problem_text_description,
    )

    # Batch process
    summary = agent.batch_process(
        limit=args.limit,
        run_final_merge_eval=not args.skip_final_eval,
        eval_n_workers=args.eval_workers,
        eval_timeout=args.eval_timeout,
    )

    # Save summary
    summary_file = agent.output_dir / 'batch_reflection_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nSummary saved to: {summary_file}")


if __name__ == '__main__':
    main()
