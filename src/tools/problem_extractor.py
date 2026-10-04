"""
Programming Problem Extraction Tool
Extracts corresponding programming problems and test cases from datasets based on flowchart names
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any


class ProblemExtractor:
    """Programming Problem Extractor"""

    def __init__(self, data_root: str = "../data"):
        """
        Initialize the problem extractor

        Args:
            data_root: Root directory path for data
        """
        self.data_root = Path(data_root)
        self.datasets = ['Algorithm', 'HumanEval-V', 'MATH', 'LiveCodeBench']
        self.dataset_cache = {}  # Cache for loaded datasets

    def extract_problem_by_image_name(self, image_name: str) -> Optional[Dict[str, Any]]:
        """
        Extract the corresponding programming problem based on flowchart image name

        Args:
            image_name: Flowchart image filename (without extension)

        Returns:
            Dictionary containing problem information, including:
            - task_id: Task ID
            - title: Problem title
            - url: Problem URL
            - test_cases: Test cases
            - meta: Metadata information
        """
        try:
            task_id = self._extract_task_id_from_image_name(image_name)
            if not task_id:
                return None

            for dataset in self.datasets:
                problem = self._find_problem_in_dataset(task_id, dataset)
                if problem:
                    return problem

            return None
        except Exception as e:
            print(f"Error extracting problem: {e}")
            return None

    def _extract_task_id_from_image_name(self, image_name: str) -> Optional[str]:
        if image_name.endswith('.png'):
            image_name = image_name[:-4]
        return image_name

    def _find_problem_in_dataset(self, task_id: str, dataset: str) -> Optional[Dict[str, Any]]:
        dataset_path = self.data_root / dataset

        if dataset == "HumanEval-V":
            jsonl_path = dataset_path / "HumanEval.jsonl"
        elif dataset == "LiveCodeBench":
            jsonl_path = dataset_path / "LiveCodeBench.jsonl"
            if not jsonl_path.exists():
                jsonl_files = list(dataset_path.glob("*.jsonl"))
                if jsonl_files:
                    jsonl_path = jsonl_files[0]
        else:
            jsonl_path = dataset_path / f"{dataset}.jsonl"

        if not jsonl_path.exists():
            if dataset == "HumanEval-V":
                jsonl_path = dataset_path / f"{dataset}.jsonl"
            else:
                return None

        if not jsonl_path.exists():
            return None

        if dataset not in self.dataset_cache:
            self.dataset_cache[dataset] = self._load_dataset(jsonl_path)

        lcb_metadata = {}
        if dataset == "LiveCodeBench":
            merged_path = dataset_path / "LiveCodeBench_merged.jsonl"
            if merged_path.exists():
                cache_key = "LiveCodeBench_merged"
                if cache_key not in self.dataset_cache:
                    self.dataset_cache[cache_key] = self._load_dataset(merged_path)

                def lcb_key(value: Any) -> str:
                    value = str(value or "")
                    return re.sub(r"^v\d+_", "", value)

                lcb_metadata = {
                    lcb_key(item.get("task_id") or item.get("question_id")): item
                    for item in self.dataset_cache[cache_key]
                    if item.get("task_id") or item.get("question_id")
                }

        for item in self.dataset_cache[dataset]:
            item_id = item.get('task_id') or item.get('question_id', '')
            if item_id == task_id:
                if lcb_metadata:
                    metadata = lcb_metadata.get(re.sub(r"^v\d+_", "", str(item_id)), {})
                    if metadata:
                        merged_item = dict(item)
                        for key in ("question_id", "question_content", "func_name", "public_test_cases", "reference_code", "difficulty"):
                            if metadata.get(key) is not None:
                                merged_item[key] = metadata[key]
                        item = merged_item
                return self._format_problem_info(item, dataset)

        return None

    def _load_dataset(self, jsonl_path: Path) -> List[Dict[str, Any]]:
        data = []
        try:
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
        except Exception as e:
            print(f"Error loading dataset file {jsonl_path}: {e}")
        return data

    def _format_problem_info(self, item: Dict[str, Any], dataset: str) -> Dict[str, Any]:
        if dataset == "LiveCodeBench" and item.get("public_test_cases"):
            test_cases = item["public_test_cases"]
        else:
            test_cases = self._extract_test_cases(item)

        problem_info = {
            'task_id': item.get('task_id', item.get('question_id', '')),
            'title': item.get('title', item.get('question_title', '')),
            'url': item.get('url', ''),
            'test_cases': test_cases,
            'starter_code': item.get('starter_code', ''),
            'text_description': self._build_problem_text_description(item),
            'meta': item.get('meta', {}),
            'dataset': dataset,
            'difficulty': item.get('meta', {}).get('difficulty', item.get('difficulty', '')),
            'category': item.get('meta', {}).get('categoryTitle', ''),
            'entry_point': item.get('entry_point', item.get('func_name', '')),
        }

        if dataset == "LiveCodeBench":
            starter_code = item.get("starter_code", "")
            func_name = item.get("func_name", "")
            is_stdio = (
                func_name == "solve"
                or (
                    starter_code.lstrip().startswith("def solve(")
                    and "class Solution" not in starter_code
                )
            )
            problem_info["lcb_mode"] = "stdio" if is_stdio else "call_based"
        if 'public_test_cases' in item:
            problem_info['public_test_cases'] = item['public_test_cases']
        if 'func_name' in item:
            problem_info['func_name'] = item['func_name']
        if 'question_content' in item:
            problem_info['question_content'] = item['question_content']
        if 'reference_code' in item:
            problem_info['reference_code'] = item['reference_code']

        return problem_info

    def _build_problem_text_description(self, item: Dict[str, Any]) -> str:
        raw_text = (
            item.get('question_content')
            or item.get('prompt_sft')
            or item.get('prompt')
            or item.get('description')
            or ""
        )
        if not isinstance(raw_text, str):
            return ""

        text = raw_text.strip()
        if text.startswith('"""') and text.endswith('"""') and len(text) >= 6:
            text = text[3:-3].strip()

        filtered_lines = []
        for line in text.splitlines():
            stripped = line.strip()
            lowered = stripped.lower()
            if lowered.startswith("assert ") or re.search(r"\\bassert\\b.*(==|\\bis\\b)", stripped):
                continue
            filtered_lines.append(line)

        text = "\\n".join(filtered_lines)
        text = re.sub(r"\\n{3,}", "\\n\\n", text).strip()
        return text

    def _extract_test_cases(self, item: Dict[str, Any]) -> List[Dict[str, Any]]:
        test_cases = []
        test_field = item.get('test', '')
        if test_field:
            test_cases = self._parse_test_field(test_field)
            if test_cases:
                return test_cases

        prompt = item.get('prompt', '')
        if prompt:
            test_cases = self._parse_examples_from_prompt(prompt)

        prompt_sft = item.get('prompt_sft', '')
        if prompt_sft and not test_cases:
            test_cases = self._parse_examples_from_prompt(prompt_sft)

        return test_cases

    def _parse_test_field(self, test_field: str) -> List[Dict[str, Any]]:
        test_cases = []
        try:
            if 'check(candidate):' in test_field or 'candidate(' in test_field:
                lines = test_field.strip().split('\\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('assert '):
                        match = re.search(r'assert candidate\\((.*?)\\)', line)
                        if match:
                            args = match.group(1)
                            test_cases.append({'input': args, 'output': None})
        except Exception as e:
            print(f"Error parsing test field: {e}")
        return test_cases

    def _parse_examples_from_prompt(self, prompt: str) -> List[Dict[str, Any]]:
        test_cases = []
        try:
            matches = re.findall(r'```python\\n(.*?)\\n```', prompt, re.DOTALL)
            for code in matches:
                test_cases.append({
                    'input': '',
                    'output': code.strip()
                })
        except Exception:
            pass
        return test_cases
