"""
Agent 1: ILR Generator with Tool Calling
Can first call the OCR tool to extract flowchart information, then generate ILR based on OCR results
"""

import json
import re
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
from tools.flowchart_cache import FlowchartCache
from agents.self_planning_prompt_utils import insert_before, insert_after
from utils.intermediate_representation import (
    get_intermediate_representation_display_name,
    normalize_intermediate_representation_type,
)


class ILRGeneratorWithTools:
    """Intermediate representation generator with tool calling capability."""

    def __init__(self, vision_model_api=None, output_dir: str = None,
                 intermediate_representation_type: str = "ilr",
                 prompt_variant: str = "default",
                 include_problem_text_description: bool = True):
        """
        Initialize the ILR generator

        Args:
            vision_model_api: Vision model API interface for generating ILR and OCR
            output_dir: Output directory path
        """
        self.vision_model_api = vision_model_api
        self.intermediate_representation_type = normalize_intermediate_representation_type(
            intermediate_representation_type
        )
        self.prompt_variant = (prompt_variant or "default").strip().lower()
        self.include_problem_text_description = include_problem_text_description
        # Ensure absolute path is used
        if output_dir:
            self.output_dir = str(Path(output_dir).resolve())
        else:
            self.output_dir = str((Path(__file__).parent.parent.parent / "output").resolve())
        self.flowchart_cache = FlowchartCache(output_dir=self.output_dir)

        # Tool definitions
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "extract_ocr_from_flowchart",
                    "description": "Extract raw text information from the flowchart image (OCR). This tool identifies all nodes, edges, and text in the flowchart and returns structured OCR data. Calling this tool before generating ILR can help accurately understand the flowchart content.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "reason": {
                                "type": "string",
                                "description": "Explanation of why the OCR tool needs to be called"
                            }
                        },
                        "required": ["reason"]
                    }
                }
            }
        ]

    def _use_self_planning(self) -> bool:
        return self.prompt_variant == "self_planning"

    def _apply_self_planning(self, prompt: str, marker: str, block: str, position: str = "before") -> str:
        if not self._use_self_planning():
            return prompt
        if position == "after":
            return insert_after(prompt, marker, block)
        return insert_before(prompt, marker, block)

    def generate_ilr_with_tools(self, image_path: str, starter_code: str = "",
                                task_id: str = None, dataset: str = None,
                                ocr_data: Dict = None,
                                problem_text_description: str = "") -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Generate ILR using tool calling (always calls OCR first)

        Workflow:
        1. Call the OCR tool first to extract text information from the flowchart
        2. Generate ILR based on OCR information and problem description

        Args:
            image_path: Original flowchart image file path
            starter_code: Interface constraints/starter code for the target function
            task_id: Task ID (used for caching OCR results)
            dataset: Dataset name (used for caching OCR results)
            ocr_data: Optional pre-extracted OCR data; if provided, the OCR step is skipped

        Returns:
            (Generated ILR, usage info dict, OCR data)
        """
        if not self.vision_model_api:
            raise ValueError("Vision model API not configured, cannot generate ILR.")

        # If task_id is not provided, extract it from image_path
        if task_id is None:
            task_id = Path(image_path).stem

        try:
            # Read image file
            with open(image_path, 'rb') as f:
                image_data = f.read()

            ocr_usage = {}

            # Step 1: Get OCR data
            # If ocr_data is provided, use it directly
            if ocr_data:
                print(f"Using provided OCR data: {task_id}")
                # Ensure it's saved to cache (if dataset exists)
                if dataset and not self.flowchart_cache.is_extracted(task_id, dataset):
                    self.flowchart_cache.save_flowchart_data(task_id, dataset, ocr_data)
            # Otherwise check cache
            elif dataset and self.flowchart_cache.is_extracted(task_id, dataset):
                print(f"Reading OCR data from cache: {task_id}")
                ocr_data = self.flowchart_cache.get_flowchart_data(task_id, dataset)
            # Otherwise call the OCR tool
            else:
                print("Calling OCR tool to extract flowchart information...")
                ocr_data, ocr_usage = self._extract_ocr(image_data)

                print(f"OCR extraction complete, identified {len(ocr_data.get('nodes', []))} nodes")

                # Save OCR data to cache
                if dataset:
                    self.flowchart_cache.save_flowchart_data(task_id, dataset, ocr_data)
                    print(f"OCR data cached: {task_id}")

            # Initialize token usage statistics
            total_usage = {
                'ocr_extraction': ocr_usage
            }

            # Step 2: Generate intermediate representation based on OCR information
            ilr_prompt = self._build_generation_prompt(
                starter_code,
                ocr_data,
                problem_text_description=problem_text_description,
            )
            ilr, ilr_usage = self.vision_model_api.analyze_image(image_data, ilr_prompt)

            # Record ILR generation token usage
            total_usage['ilr_generation'] = ilr_usage

            # Validate the returned ILR
            if not ilr or not ilr.strip():
                display_name = get_intermediate_representation_display_name(
                    self.intermediate_representation_type
                )
                raise ValueError(f"Model returned empty {display_name}")

            ilr_stripped = ilr.strip()
            if ilr_stripped.startswith("//") and "failed" in ilr_stripped.lower():
                raise ValueError(f"Model returned an error message: {ilr_stripped}")

            if ilr_stripped.startswith("Error"):
                raise ValueError(f"Model returned an error message: {ilr_stripped}")

            return ilr, total_usage, ocr_data

        except Exception as e:
            print(f"Error during ILR generation: {e}")
            raise

    def generate_ilr_with_reflection(self, image_path: str, reflection_prompt: str,
                                     starter_code: str = "", task_id: str = None,
                                     dataset: str = None, ocr_data: Dict = None,
                                     original_ilr: str = None,
                                     problem_text_description: str = "") -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Regenerate ILR based on reflection prompt

        Args:
            image_path: Original flowchart image file path
            reflection_prompt: Reflection analysis and improvement suggestions
            starter_code: Interface constraints/starter code for the target function
            task_id: Task ID (used for reading cached OCR results)
            dataset: Dataset name (used for reading cached OCR results)
            ocr_data: Optional pre-extracted OCR data; if provided, the OCR step is skipped
            original_ilr: Original problematic ILR string

        Returns:
            (Generated ILR, usage info dict, OCR data)
        """
        if not self.vision_model_api:
            raise ValueError("Vision model API not configured, cannot generate ILR.")

        # If task_id is not provided, extract it from image_path
        if task_id is None:
            task_id = Path(image_path).stem

        try:
            # Read image file
            with open(image_path, 'rb') as f:
                image_data = f.read()

            ocr_usage = {}

            # Step 1: Get OCR data
            if ocr_data:
                print(f"Using provided OCR data (reflection mode): {task_id}")
            elif dataset and self.flowchart_cache.is_extracted(task_id, dataset):
                print(f"Reading OCR data from cache (reflection mode): {task_id}")
                ocr_data = self.flowchart_cache.get_flowchart_data(task_id, dataset)
            else:
                # If not in cache, extract OCR (should not normally happen)
                print("Warning: OCR data not found in cache, re-extracting...")
                ocr_data, ocr_usage = self._extract_ocr(image_data)
                print(f"OCR extraction complete, identified {len(ocr_data.get('nodes', []))} nodes")

                # Save OCR data to cache
                if dataset:
                    self.flowchart_cache.save_flowchart_data(task_id, dataset, ocr_data)

            # Initialize token usage statistics
            total_usage = {
                'ocr_extraction': ocr_usage
            }

            # Step 2: Generate improved intermediate representation based on OCR info and reflection prompt
            ilr_prompt = self._build_generation_prompt_with_reflection(
                starter_code,
                ocr_data,
                reflection_prompt,
                original_ilr,
                problem_text_description=problem_text_description,
            )
            ilr, ilr_usage = self.vision_model_api.analyze_image(image_data, ilr_prompt)

            # Record ILR generation token usage
            total_usage['ilr_generation'] = ilr_usage

            # Validate the returned ILR
            if not ilr or not ilr.strip():
                display_name = get_intermediate_representation_display_name(
                    self.intermediate_representation_type
                )
                raise ValueError(f"Model returned empty {display_name}")

            ilr_stripped = ilr.strip()
            if ilr_stripped.startswith("//") and "failed" in ilr_stripped.lower():
                raise ValueError(f"Model returned an error message: {ilr_stripped}")

            if ilr_stripped.startswith("Error"):
                raise ValueError(f"Model returned an error message: {ilr_stripped}")

            return ilr, total_usage, ocr_data

        except Exception as e:
            print(f"Error during reflection-based ILR generation: {e}")
            raise

    def _call_with_tools(self, image_data: bytes, prompt: str) -> Tuple[str, Dict]:
        """
        Call the API that supports tools

        Note: This assumes vision_model_api supports tool calling.
        If not supported, a normal response will be returned directly.
        """
        # Check if the API supports tool calling
        if hasattr(self.vision_model_api, 'analyze_image_with_tools'):
            return self.vision_model_api.analyze_image_with_tools(
                image_data, prompt, self.tools
            )
        else:
            # If tool calling is not supported, return normal response directly
            return self.vision_model_api.analyze_image(image_data, prompt)

    def _should_call_ocr(self, response: str) -> bool:
        """
        Determine whether the agent's response indicates a need to call the OCR tool

        Checks if the response contains markers for tool calling
        """
        # Check if it contains tool calling JSON format
        if "extract_ocr_from_flowchart" in response:
            return True

        # Check if it contains explicit OCR request keywords
        ocr_keywords = [ "need ocr", "call ocr"]
        response_lower = response.lower()
        return any(keyword.lower() in response_lower for keyword in ocr_keywords)

    def _extract_ocr(self, image_data: bytes) -> Tuple[Dict[str, Any], Dict]:
        """
        Perform OCR extraction

        Returns:
            (OCR data, token usage info)
        """
        ocr_prompt = self._build_ocr_prompt()
        response, usage = self.vision_model_api.analyze_image(image_data, ocr_prompt)

        # Parse OCR response
        ocr_data = self._parse_ocr_response(response)

        return ocr_data, usage

    def _build_interface_section(self, starter_code: str) -> str:
        """Build interface constraints description."""
        if not starter_code or not starter_code.strip():
            return ""
        return f"""
## Starter Code / Interface
Use the following starter code as the ONLY external interface constraint:

```python
{starter_code}
```

- Preserve the function name and parameter order implied by this starter code.
"""

    def _build_problem_text_section(self, problem_text_description: str = "") -> str:
        """Build problem text description section."""
        if not self.include_problem_text_description:
            return ""

        description = (problem_text_description or "").strip()
        if not description:
            return ""

        return f"""
## Problem Text Description
The following dataset text describes the programming task.
Use it as auxiliary context for the intended algorithm and edge cases.
The flowchart image remains the primary source for control flow and node semantics.
Starter code remains the authoritative interface contract.

{description}
"""

    def _is_stdio_interface(self, starter_code: str = "", description: str = "") -> bool:
        """Determine the LCB standard-input interface from starter code."""
        # Keep description for API compatibility; problem text does not affect mode detection.
        del description
        starter = (starter_code or "").lstrip()
        return re.match(r"def\s+solve\s*\(", starter) is not None

    def _build_io_contract_section(self, starter_code: str = "", description: str = "") -> str:
        """Add parameterized ILR constraints for standard-input tasks."""
        if not self._is_stdio_interface(starter_code, description):
            return ""
        return """
## LCB Standard-Input Contract
- Treat `arg0`, `arg1`, `arg2` ... as already parsed input values.
- Initialize these values from `ctx['arg0']`, `ctx['arg1']`, ... in the original order.
- Do not put `input()`, `sys.stdin`, `print()`, or file I/O in ILR actions or logic.
- Represent the answer with the final return/output node.
"""
    def _build_initial_prompt(self, starter_code: str = "") -> str:
        """Build initial observation prompt"""
        prompt = f"""# Task: Analyze Flowchart and Decide Strategy

You are an expert at converting flowcharts to executable logic representations (ILR).

## Your Task:
1. Observe the flowchart image
2. Decide if you need to call the OCR tool to extract text information first
3. Explain your reasoning

## Available Tools:
- extract_ocr_from_flowchart: Extract all text and structure from the flowchart

## When to Use OCR:
- If the flowchart is complex with many nodes
- If text in the flowchart is small or unclear
- If you want to ensure accurate text extraction before generating ILR
{self._build_interface_section(starter_code)}

## Your Response:
Please analyze the flowchart and decide:
1. Do you need to call extract_ocr_from_flowchart first? (Yes/No)
2. Why or why not?

If you decide to use OCR, respond with:
{{
  "tool_call": "extract_ocr_from_flowchart",
  "reason": "your reason here"
}}

Otherwise, respond with:
{{
  "tool_call": "none",
  "reason": "your reason here"
}}
"""
        return self._apply_self_planning(
            prompt,
            "## Your Response:",
            "## Self-Planning (internal)\n- Inspect complexity and text clarity\n- Decide OCR usage based on ambiguity\nDo NOT output the plan."
        )

    def _build_ocr_prompt(self) -> str:
        """Build OCR extraction prompt"""
        prompt = """# Task: Pure OCR Text Extraction from Flowchart

## Objective:
Extract ALL visible text from the flowchart image EXACTLY as it appears.

## Output Format:
Return a JSON object:

{
    "nodes": [
        {
            "node_id": 1,
            "shape_type": "oval/rectangle/diamond/parallelogram",
            "raw_text": "exact text inside the shape",
            "position": "top/middle/bottom"
        }
    ],
    "edges": [
        {
            "from_node": 1,
            "to_node": 2,
            "label": "text on arrow (if any)"
        }
    ],
    "other_text": ["any other text"]
}

## Rules:
- Extract text EXACTLY as shown
- Do NOT interpret or modify text
- List nodes from top to bottom, left to right
- Output ONLY JSON, no markdown
"""
        return self._apply_self_planning(
            prompt,
            "## Output Format:",
            "## Self-Planning (internal)\n- Scan nodes and edges systematically\n- Preserve exact text verbatim\nDo NOT output the plan."
        )

    def _build_ilr_generation_prompt(self, starter_code: str = "",
                                     ocr_data: Optional[Dict] = None,
                                     problem_text_description: str = "") -> str:
        """Build ILR generation prompt"""
        base_prompt = """# Task: Convert flowchart to ELM (Executable Logic Manifest) JSON

## CRITICAL Checklist (Verify before output):
- MANDATORY: First process node MUST initialize ALL variables used in the flowchart
- **PARAMETER ORDER**: Carefully match arg0, arg1, arg2... to problem description parameter order and types
- All variables initialized BEFORE use (especially loop counters: i=0)
- **SYNTAX**: All strings use matching quotes, all code blocks properly indented, no incomplete statements
- Main algorithm loops use decision nodes; Simple initialization for loops OK
- Only use allowed built-in functions (see Python Environment section)
- Output format matches problem description (tuple: (a,b), list: [x], single: x)
- If flowchart has subgraphs (helper functions), include them in functions array

## Schema:
{
  "problem_id": "string",
  "functions": [{"name": "func", "parameters": ["a"], "entry_node": 1, "nodes": [...]}],  // Optional
  "nodes": [...]  // Main function nodes
}

## Node Types:
1. start: {id, type, label, bbox, next}
   - label: "Start" or original text inside shape
2. process: {id, type, label, bbox, action, next}
   - label: ORIGINAL text from flowchart (e.g., "Add 1 to i")
   - action: Python code (e.g., "ctx['i'] += 1")
3. decision: {id, type, label, bbox, logic, true_next, false_next}
   - label: ORIGINAL text from flowchart (e.g., "Is i < n?")
   - logic: Python expression (e.g., "ctx['i'] < ctx['n']")
4. call: {id, type, label, bbox, function_name, arguments, return_var, next}
   - label: ORIGINAL text
5. return: {id, type, label, bbox, value}
   - label: ORIGINAL text
6. end: {id, type, label, bbox, output}
   - label: ORIGINAL text

## Key Rules:

### 1. Label Integrity (CRITICAL!)
- **MANDATORY**: The `label` field MUST contain the **EXACT ORIGINAL TEXT** from the flowchart node.
- **DO NOT** convert natural language to code in the `label` field.
- **DO NOT** fix typos or abbreviations in the `label` field.
- **ONLY** perform code conversion in the `action` (for process) or `logic` (for decision) fields.

Example:
- Flowchart text: "Set count to 0"
- **WRONG**: {"label": "ctx['count'] = 0", "action": "ctx['count'] = 0"}
- **CORRECT**: {"label": "Set count to 0", "action": "ctx['count'] = 0"}

### 2. Parameter Order and Type Matching (CRITICAL!)
**MOST COMMON ERROR**: Mismatching parameter order between problem description and ILR initialization.

VERIFICATION STEPS:
1. Identify from problem description:
   - Parameter names (e.g., "nums", "k", "target")
   - Parameter types (array, number, string, etc.)
   - Parameter order (first, second, third...)

2. Map correctly:
   - First parameter → ctx['arg0']
   - Second parameter → ctx['arg1']
   - Third parameter → ctx['arg2']

3. Examples:
   - "findKthLargest(nums, k)" → ctx['nums']=ctx['arg0']; ctx['k']=ctx['arg1']
   - "twoSum(target, nums)" → ctx['target']=ctx['arg0']; ctx['nums']=ctx['arg1']

**WRONG**: ctx['k']=ctx['arg0']; ctx['nums']=ctx['arg1']  ❌ Order reversed!
**CORRECT**: ctx['nums']=ctx['arg0']; ctx['k']=ctx['arg1']  ✓

### Syntax Rules (CRITICAL!)

1. **String Quotes**: Always match quotes
   - WRONG: ctx['s'][-1']  ❌
   - CORRECT: ctx['s'][-1]  ✓

2. **Function Definitions**: Must have body
   - WRONG: def dp(pos, tight):  ❌ Missing body
   - CORRECT: def dp(pos, tight): return 0  ✓

3. **For Loop**: Single-line OR multi-line with indentation
   - Single-line: for v in ctx['nums']: ctx['count'][v] = 1  ✓
   - Multi-line: for v in ctx['nums']:\n    ctx['count'][v] = 1  ✓
   - CRITICAL: Loop body MUST be indented (4 spaces) after newline

4. **If-Else**: Use ternary OR multi-line with indentation
   - Ternary: ctx['result'] += char.upper() if char.islower() else char.lower()  ✓
   - Multi-line: if char.islower():\n    ctx['result'] += char.upper()\nelse:\n    ctx['result'] += char.lower()  ✓
   - CRITICAL: if/else body MUST be indented after newline

5. **Indentation**: Use 4 spaces (or 1 tab) for each level
   - MUST indent after: for, while, if, else, elif, def
   - ❌ Missing indentation after colon
   - ❌ Mixing spaces and tabs

### Variable Initialization (CRITICAL!)
**MANDATORY**: First node after start MUST initialize ALL variables.

Pattern:
```json
{"id": 1, "type": "start", "next": 2},
{"id": 2, "type": "process", "action": "ctx['n']=ctx['arg0']; ctx['i']=0; ctx['sum']=0", "next": 3}
```

Initialization types:
- Input: ctx['n'] = ctx['arg0']
- Counters: ctx['i'] = 0
- Accumulators: ctx['sum'] = 0
- Lists: ctx['result'] = []
- Dicts: ctx['map'] = {}
- Booleans: ctx['found'] = False

### Loop Structure
- MAIN algorithm loops: Use decision nodes (for flowchart visibility)
- SIMPLE initialization: May use for loops in action (for conciseness)

ALLOWED: {"action": "ctx['hw'] = {}; for v in ctx['vowels']: ctx['hw'][v] = True"}
NOT ALLOWED: {"action": "for i in range(n): if condition: complex_logic()"}  Use decision nodes!

### Output Format
- Single: "ctx['result']"
- Tuple: "(ctx['count'], ctx['message'])"  Use parentheses!
- List: "ctx['result_list']"

### Python Environment
Available: len, range, int, float, str, bool, list, dict, set, tuple, abs, min, max, sum, sorted, reversed, enumerate, zip, map, filter, all, any, round, pow, divmod, ord, chr, isinstance, type, iter
NOT available: next, lambda, bin, hex, oct, imports, file I/O

### Helper Functions (if flowchart has subgraphs)
Structure:
```json
"functions": [{
  "name": "helper_name",
  "parameters": ["param1"],
  "entry_node": 1,
  "nodes": [
    {"id": 1, "type": "start", "next": 2},
    {"id": 2, "type": "process", "action": "ctx['var']=...", "next": 3},
    {"id": N, "type": "return", "value": "ctx['result']"}
  ]
}]
```

Call: {"type": "call", "function_name": "helper_name", "arguments": ["ctx['x']"], "return_var": "result", "next": 6}

## Common Mistakes:
- **#1**: Wrong parameter order/type (e.g., swapping arg0 and arg1) - VERIFY!
- **#2**: Syntax errors (mismatched quotes, incomplete functions, wrong indentation)
- **#3**: Missing initialization node after start
- Missing indentation after for/if/else/while
- Incomplete function definition without body
- For loop without proper structure (single-line OR multi-line with indent)
- Uninitialized variables (especially loop counters)
- Using for loops for MAIN algorithm logic (use decision nodes!)
- Wrong output format (tuple needs parentheses)
- Forgetting to increment loop counter
- Multiple statements without semicolon (when on same line)

## Output: Pure JSON only, no markdown.

Problem Description:
"""
        base_prompt = self._apply_self_planning(
            base_prompt,
            "## Output: Pure JSON only, no markdown.",
            "## Self-Planning (internal)\n- Map parameters to ctx['arg0'], ctx['arg1']... in order\n- Outline node sequence and control flow\n- Verify checklist and output format\nDo NOT output the plan."
        )

        # If OCR data is available, add it to the prompt
        if ocr_data:
            ocr_section = f"""
## OCR Extracted Information:

The following text and structure were extracted from the flowchart:

```json
{json.dumps(ocr_data, indent=2, ensure_ascii=False)}
```

Use this OCR information as reference to ensure accurate text extraction.
Pay special attention to:
- Node labels and their exact text
- Edge labels (Yes/No, True/False, etc.)
- Any mathematical symbols or operators

"""
            base_prompt = ocr_section + base_prompt

        # Add problem description
        base_prompt += f"\n\n{problem_text_description}"
        base_prompt += self._build_io_contract_section(starter_code, problem_text_description)

        # Apply self_planning (preserve flowchart version specific functionality)
        base_prompt = self._apply_self_planning(
            base_prompt,
            "## Output: Pure JSON only, no markdown.",
            "## Self-Planning (internal)\n- Map parameters to ctx['arg0'], ctx['arg1']... in order\n- Outline node sequence and control flow\n- Verify checklist and output format\nDo NOT output the plan."
        )

        return base_prompt

    def _build_text_intermediate_generation_prompt(self, starter_code: str = "",
                                                   ocr_data: Optional[Dict] = None,
                                                   problem_text_description: str = "") -> str:
        """Build text intermediate representation generation prompt."""
        base_prompt = """# Task: Convert flowchart to Textual Intermediate Representation

You are running an ablation setting where Agent 1 must NOT output ILR JSON.
Instead, generate a plain-text intermediate representation that preserves the flowchart logic for Agent 2.

## Goal
Describe the algorithm in clear natural language so another model can write code from it.

## Hard Constraints
- DO NOT output JSON
- DO NOT output Python code
- DO NOT use markdown code fences
- Preserve the original control flow faithfully
- Explicitly state parameter mapping in the correct order: arg0, arg1, arg2...
- Explicitly state every initialization step before the variable is used
- Explicitly mark loops and branches using words like IF / ELSE / WHILE / FOR EACH / RETURN
- If the flowchart contains helper subgraphs, describe them clearly (no fixed section names required)
- Use starter code only as interface guidance

## Writing Rules
- Prefer concise, executable descriptions of the logic
- Mention important variables by name consistently
- Preserve branch meaning (Yes/No, True/False) accurately
- Preserve return value semantics accurately
- Keep helper function calls explicit
- Keep the final answer plain text only
- No fixed format is required; you may use short paragraphs or bullet points
"""
        base_prompt = self._apply_self_planning(
            base_prompt,
            "## Writing Rules",
            "## Self-Planning (internal)\n- Outline control flow and initialization order\n- Ensure parameter mapping is explicit and correct\n- Identify key branches and loops\nDo NOT output the plan."
        )

        if ocr_data:
            ocr_section = f"""
## OCR Extracted Information:

The following text and structure were extracted from the flowchart:

{json.dumps(ocr_data, indent=2, ensure_ascii=False)}

Use this OCR information as reference to ensure accurate textual logic reconstruction.
Pay special attention to node text, branch labels, and mathematical symbols.

"""
            base_prompt = ocr_section + base_prompt

        base_prompt += self._build_problem_text_section(problem_text_description)
        base_prompt += self._build_interface_section(starter_code)
        base_prompt += self._build_io_contract_section(starter_code, problem_text_description)
        return base_prompt

    def _build_generation_prompt(self, starter_code: str = "",
                                 ocr_data: Optional[Dict] = None,
                                 problem_text_description: str = "") -> str:
        """Build intermediate representation generation prompt based on configuration."""
        if self.intermediate_representation_type == "text":
            return self._build_text_intermediate_generation_prompt(
                starter_code,
                ocr_data,
                problem_text_description,
            )
        return self._build_ilr_generation_prompt(
            starter_code,
            ocr_data,
            problem_text_description,
        )

    def _build_ilr_generation_prompt_with_reflection(self, starter_code: str = "",
                                                     ocr_data: Optional[Dict] = None,
                                                     reflection_prompt: str = "",
                                                     original_ilr: str = None,
                                                     problem_text_description: str = "") -> str:
        """Build ILR generation prompt with reflection (following the structure of the normal generation prompt)"""

        # Use the same base prompt as normal generation
        base_prompt = self._build_ilr_generation_prompt(
            starter_code,
            ocr_data,
            problem_text_description=problem_text_description,
        )

        # Add reflection section
        reflection_section = f"""

## Previous Attempt and Reflection

We previously generated an ILR that failed test cases.

### Previous ILR (INCORRECT):
```json
{original_ilr if original_ilr else "N/A"}
```

### Issues Identified:
{reflection_prompt}

### Task:
Before generating the corrected ILR, briefly trace through the previous ILR's execution flow for a failing test case to pinpoint where the logic goes wrong. Keep this reasoning concise (3-5 sentences max), then generate the CORRECTED ILR.
Follow all the rules and guidelines above.
Output pure JSON only, no markdown.
"""
        if self._use_self_planning():
            reflection_section = self._apply_self_planning(
                reflection_section,
                "### Task:",
                "Self-Planning (internal): briefly plan the corrections and verification steps. Do NOT output the plan.",
                position="after"
            )

        return base_prompt + reflection_section

    def _build_text_intermediate_generation_prompt_with_reflection(
        self,
        starter_code: str = "",
        ocr_data: Optional[Dict] = None,
        reflection_prompt: str = "",
        original_ilr: str = None,
        problem_text_description: str = ""
    ) -> str:
        """Build text intermediate representation prompt with reflection."""
        base_prompt = self._build_text_intermediate_generation_prompt(
            starter_code,
            ocr_data,
            problem_text_description=problem_text_description,
        )
        reflection_section = f"""

## Previous Attempt and Reflection

We previously generated a textual intermediate representation that failed downstream testing.

### Previous Text IR (INCORRECT):
{original_ilr if original_ilr else "N/A"}

### Issues Identified:
{reflection_prompt}

### Task:
Before generating the corrected textual intermediate representation, briefly trace through the previous representation for a failing case and identify where the logic went wrong.
Then produce the corrected Text IR using clear natural language (no fixed format).
Do not output JSON or Python code.
"""
        if self._use_self_planning():
            reflection_section = self._apply_self_planning(
                reflection_section,
                "### Task:",
                "Self-Planning (internal): briefly plan the corrections and verification steps. Do NOT output the plan.",
                position="after"
            )
        return base_prompt + reflection_section

    def _build_generation_prompt_with_reflection(self, starter_code: str = "",
                                                 ocr_data: Optional[Dict] = None,
                                                 reflection_prompt: str = "",
                                                 original_ilr: str = None,
                                                 problem_text_description: str = "") -> str:
        """Build intermediate representation generation prompt with reflection based on configuration."""
        if self.intermediate_representation_type == "text":
            return self._build_text_intermediate_generation_prompt_with_reflection(
                starter_code,
                ocr_data,
                reflection_prompt,
                original_ilr,
                problem_text_description,
            )
        return self._build_ilr_generation_prompt_with_reflection(
            starter_code,
            ocr_data,
            reflection_prompt,
            original_ilr,
            problem_text_description,
        )

    def _parse_ocr_response(self, response: str) -> Dict[str, Any]:
        """Parse OCR response"""
        try:
            # Try to parse JSON directly
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON part
            import re
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass

            # If failed, return empty structure
            return {
                "nodes": [],
                "edges": [],
                "other_text": [],
                "parse_error": True,
                "raw_response": response
            }

    # Backward-compatible simple interface
    def generate_ilr(self, image_path: str, starter_code: str = "", task_id: str = None,
                     dataset: str = None,
                     problem_text_description: str = "") -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        """
        Simple interface: Generate ILR (always calls OCR first)

        This method maintains an interface similar to the original ILRGenerator, but returns additional OCR data

        Args:
            image_path: Image path
            starter_code: Interface constraints/starter code for the target function
            task_id: Task ID (optional)
            dataset: Dataset name (optional)

        Returns:
            (ILR string, token usage info, OCR data)
        """
        return self.generate_ilr_with_tools(
            image_path,
            starter_code=starter_code,
            task_id=task_id,
            dataset=dataset,
            problem_text_description=problem_text_description,
        )


def test_ilr_generator_with_tools():
    """Test the ILR generator with tools"""
    from utils.vision_api_client import VisionAPIClient

    # Load config
    config_path = Path(__file__).parent.parent / "configs" / "qwen2_5_vl_7b_local_config.json"
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create API client
    api_client = VisionAPIClient(config_dict=config)

    # Create ILR generator
    generator = ILRGeneratorWithTools(vision_model_api=api_client)

    # Test ILR generation
    test_image_path = "test_image.png"
    starter_code = "def solution(word):"

    print("Starting ILR generation (with tool calling)...")
    ilr, usage = generator.generate_ilr(test_image_path, starter_code=starter_code)

    print("\nGenerated ILR:")
    print(ilr)
    print(f"\nToken usage: {usage}")


if __name__ == "__main__":
    test_ilr_generator_with_tools()
