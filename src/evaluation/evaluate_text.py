"""
Text-to-Code Evaluation Script
==============================
Input: dataset JSONL (natural language prompt + starter_code)
Model: VLM API (text mode, no images)
Output: samples.jsonl -> evaluation -> results.txt (same format as evaluate.py)
"""

import re
import argparse
import json
import os
import time
import requests
from tqdm import tqdm
from openai import OpenAI

# ── File I/O ────────────────────────────────────────────────────────────────

def read_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                results.append(json.loads(line.strip()))
            except:
                continue
    return results

def write_jsonl_file(file_path, data):
    with open(file_path, 'w', encoding='utf-8') as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')

# ── Code Extraction ─────────────────────────────────────────────────────────

def extract_code(content):
    """Extract Python code from model response"""
    try:
        if '```python' in content:
            p_code = re.compile(r'```python\n(.*?)\n```', flags=re.DOTALL)
            code_block = p_code.findall(content)[0]
            if "assert" in code_block:
                code_block = code_block.split("assert")[0]
            return code_block
        elif '```' in content:
            p_code = re.compile(r'```(.*?)\n(.*?)```', flags=re.DOTALL)
            code_block = p_code.findall(content)[0][1]
            if "assert" in code_block:
                code_block = code_block.split("assert")[0]
            return code_block
        else:
            return content
    except Exception:
        return content

# ── Prompt Construction ─────────────────────────────────────────────────────

def _is_stdin_problem(problem):
    """Detect whether the problem expects stdin/stdout (AtCoder style) vs call-based (LeetCode style)."""
    # If the test cases use "assert solve(" format, it's call-based regardless of prompt content
    test_str = problem.get("test", "")
    if test_str and "assert solve(" in test_str:
        return False
    # AtCoder problems mention "Standard Input" in the problem description
    prompt_text = problem.get("prompt", "")
    if "Standard Input" in prompt_text or "standard input" in prompt_text:
        return True
    # Also check starter_code for solve() signature without class
    starter = problem.get("starter_code", "")
    if starter and "def solve(" in starter and "class Solution" not in starter:
        return True
    return False


def construct_prompt_text(problem):
    """
    Build prompt from problem description.
    The prompt field already contains the full problem description (including docstring and class/method signature),
    so we append output instructions directly without repeating starter_code.
    """
    prompt_text = problem.get("prompt", "").strip()
    is_stdin = _is_stdin_problem(problem)

    if is_stdin:
        prompt = f"""{prompt_text}

Note:
- If you need to import any Python package, import it yourself.
- Read input from stdin using input() or sys.stdin.readline().
- Print the answer using print().
- The function solve() will be called automatically; you do NOT need to call solve() yourself.
- The function parameters are provided for reference; prefer reading from stdin inside the function.
- Output ONLY valid ASCII Python code (no non-ASCII characters).
- Do NOT include test cases, assertions, or if __name__ == '__main__' blocks.
- Write your code between ```python and ``` delimiters.

Complete the code based on the problem description above.
"""
    else:
        prompt = f"""{prompt_text}

Note:
- If you need to import any Python package, import it yourself.
- Do NOT use input() to read input — just complete the function.
- The function should RETURN the result, not print it.
- Output ONLY valid ASCII Python code (no non-ASCII characters).
- Do NOT include test cases, assertions, or if __name__ == '__main__' blocks.
- Write your code between ```python and ``` delimiters.

Complete the code based on the problem description above.
"""
    return prompt

def construct_prompt_cot_text(problem):
    """Chain-of-Thought version"""
    prompt_text = problem.get("prompt", "").strip()
    is_stdin = _is_stdin_problem(problem)

    stdin_note = (
        "- Read input from stdin using input() or sys.stdin.readline().\n"
        "- Print the answer using print().\n"
        "- The function solve() will be called automatically; you do NOT need to call solve() yourself.\n"
        if is_stdin else
        "- Do NOT use input() to read input — just complete the function.\n"
        "- The function should RETURN the result, not print it.\n"
    )

    prompt = f"""{prompt_text}

Before writing the code, think step by step about the algorithm.

Rules:
- Put your reasoning outside the code block.
- The code block must contain only Python code.
- If you need to import any package, import it yourself.
{stdin_note}- Output ONLY valid ASCII Python code (no non-ASCII characters).
- Do NOT include test cases, assertions, or if __name__ == '__main__' blocks.

Complete the code based on the problem description above.
"""
    return prompt

# ── API Calls ────────────────────────────────────────────────────────────────

def _build_model_candidates(api_config):
    candidates = []
    model_name = str(api_config.get("model", "")).strip()
    if model_name:
        candidates.append(model_name)
    for alias in api_config.get("model_aliases", []):
        alias_str = str(alias).strip()
        if alias_str:
            candidates.append(alias_str)
    if "/" in model_name:
        basename = model_name.split("/")[-1].strip()
        if basename:
            candidates.append(basename)
            candidates.append(basename.lower())
        candidates.append(model_name.lower())
    dedup = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            dedup.append(item)
            seen.add(item)
    return dedup

def get_response_text(prompt, api_config, max_retries=3, retry_delay=5):
    """
    Pure text API call (no images).
    Returns (responses, usage, model_name, total_retry_time)
    """
    base_url = api_config["base_url"].rstrip('/')
    api_key = api_config["api_key"]

    # Support directly providing the full API endpoint path, e.g. .../v1/messages
    if api_config.get("use_direct_messages_url", False) or base_url.endswith("/v1/messages"):
        url = base_url
    elif base_url.endswith("/v1") or base_url.endswith("/v4"):
        url = base_url + "/chat/completions"
    else:
        url = base_url + "/v1/chat/completions"

    model_candidates = _build_model_candidates(api_config)
    if not model_candidates:
        model_candidates = [api_config.get("model", "")]

    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    rate_limit_retries = 0
    max_rate_limit_retries = 3
    total_retry_time = 0

    for model_idx, model_candidate in enumerate(model_candidates):
        if model_idx > 0:
            print(f"Trying model alias: {model_candidate}")

        payload_dict = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ],
            "max_tokens": api_config["max_tokens"],
            "temperature": api_config["temperature"],
            "n": api_config["n"]
        }

        # Only add 'stop' parameter if it's provided and non-empty
        if api_config.get("stop"):
            payload_dict["stop"] = api_config["stop"]

        if "top_p" in api_config:
            payload_dict["top_p"] = api_config["top_p"]
        if "thinking" in api_config:
            payload_dict["thinking"] = api_config["thinking"]
        if "thinking_mode" in api_config:
            payload_dict["thinking_mode"] = api_config["thinking_mode"]

        # Streaming output: add stream parameter to payload and print content in real-time
        if api_config.get("stream", False):
            payload_dict["stream"] = True
            payload_dict["stream_options"] = {"include_usage": True}

        payload_dict["model"] = model_candidate
        payload = json.dumps(payload_dict, ensure_ascii=False)

        switch_to_next_model = False
        for attempt in range(max_retries):
            try:
                timeout = api_config.get("timeout", 600)
                if api_config.get("stream", False):
                    response = requests.post(url, headers=headers, json=payload_dict, timeout=timeout, stream=True)
                else:
                    response = requests.post(url, headers=headers, json=payload_dict, timeout=timeout)

                if response.status_code != 200:
                    error_text = response.text[:500]

                    if response.status_code == 429:
                        rate_limit_retries += 1
                        if rate_limit_retries <= max_rate_limit_retries:
                            print(f"Encountered 429 rate limit error (attempt {rate_limit_retries}/{max_rate_limit_retries}), waiting 30 seconds...")
                            time.sleep(30)
                            total_retry_time += 30
                            continue
                        else:
                            raise Exception(f"API request failed: 429 Rate Limit Exceeded")

                    has_next_model = model_idx < len(model_candidates) - 1
                    lower_error = error_text.lower()
                    if (has_next_model and response.status_code in (400, 404)
                            and "model" in lower_error and ("not found" in lower_error or "does not exist" in lower_error)):
                        print(f"Model name {model_candidate} may not be registered on the server, trying next alias...")
                        switch_to_next_model = True
                        break

                    print(f"API request failed: {response.status_code}, response: {error_text}")
                    raise Exception(f"API request failed: {response.status_code}")

                if not response.text.strip() and not api_config.get("stream", False):
                    raise Exception("API returned empty response")

                # Streaming output: print to terminal in real-time
                if api_config.get("stream", False):
                    full_content = ""
                    completion_tokens = 0
                    import sys
                    stream_label = f"[stream][{model_candidate}]"
                    try:
                        for line in response.iter_lines():
                            if not line:
                                continue
                            try:
                                line_text = line.decode("utf-8")
                            except UnicodeDecodeError:
                                continue
                            if line_text.startswith("data: "):
                                data = line_text[6:]
                                if data == "[DONE]":
                                    break
                                try:
                                    delta = json.loads(data)
                                    # Handle different stream formats
                                    choices = delta.get("choices", [])
                                    if choices and len(choices) > 0:
                                        choice = choices[0]
                                        chunk = choice.get("delta", {})
                                        content_piece = chunk.get("content", "") or chunk.get("text", "")
                                        if content_piece:
                                            full_content += content_piece
                                            try:
                                                sys.stdout.write(content_piece)
                                                sys.stdout.flush()
                                            except (UnicodeEncodeError, OSError):
                                                # Windows console encoding issue or no console, skip printing
                                                pass
                                    usage = delta.get("usage", {})
                                    if usage and isinstance(usage, dict):
                                        completion_tokens = usage.get("completion_tokens", completion_tokens)
                                except (json.JSONDecodeError, IndexError, KeyError, TypeError):
                                    continue
                        try:
                            sys.stdout.write(f"\n{stream_label}\n")
                            sys.stdout.flush()
                        except (UnicodeEncodeError, OSError):
                            pass
                    except Exception as e:
                        print(f"[WARNING] Stream processing error: {e}")
                    responses = [full_content]
                    usage = {
                        "completion_tokens": completion_tokens,
                        "prompt_tokens": 0,
                        "total_tokens": completion_tokens,
                    }
                    return responses, usage, model_candidate, total_retry_time

                # Non-streaming output
                response_json = response.json()

                responses = []
                for i in range(api_config["n"]):
                    message = response_json["choices"][i]["message"]
                    content = message.get("content", "")
                    if not content or not content.strip():
                        print(f"Warning: content of response {i} is empty")
                    responses.append(content)

                usage = response_json.get("usage", {})
                if isinstance(usage, dict):
                    usage["max_completion_tokens"] = usage.get("completion_tokens", 0)

                for i, choice in enumerate(response_json.get("choices", [])):
                    if choice.get("finish_reason") == "length":
                        print(f"Warning: response {i} was truncated due to max_tokens limit")

                return responses, usage, response_json.get("model", model_candidate), total_retry_time

            except Exception as e:
                if "429" in str(e) and rate_limit_retries <= max_rate_limit_retries:
                    continue
                if switch_to_next_model:
                    break
                print(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                    total_retry_time += retry_delay
                else:
                    break

    print(f"All model candidates exhausted.")
    return ["ERROR: API request failed"], None, api_config.get("model", ""), total_retry_time

# ── Evaluation & Scoring ─────────────────────────────────────────────────

def calculate_token_statistics(results):
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    valid_count = 0

    for result in results:
        usage = result.get("usage", {})
        if isinstance(usage, dict) and usage:
            total_prompt_tokens += usage.get("prompt_tokens", 0)
            total_completion_tokens += usage.get("completion_tokens", 0)
            total_tokens += usage.get("total_tokens", 0)
            valid_count += 1

    if valid_count > 0:
        return {
            "total_tasks": valid_count,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_prompt_tokens": total_prompt_tokens / valid_count,
            "avg_completion_tokens": total_completion_tokens / valid_count,
            "avg_total_tokens": total_tokens / valid_count
        }
    return None

def run_evaluation(sample_file, problem_file):
    """Run human-eval evaluation"""
    cmd = f"evaluate_functional_correctness {sample_file} --problem_file={problem_file}"
    print(f"Running: {cmd}")
    os.system(cmd)

def score_results(results_file, output_path):
    """Calculate pass rate and write results.txt"""
    results = read_jsonl_file(results_file)

    keys = results[0].keys()
    easy, medium, hard = [], [], []

    if "meta" in keys:
        for line in results:
            d = line["meta"].get("difficulty", "").lower()
            if d == "easy":
                easy.append(line)
            elif d == "medium":
                medium.append(line)
            elif d == "hard":
                hard.append(line)
    elif "difficulty" in keys:
        for line in results:
            d = line.get("difficulty", "").lower()
            if d == "easy":
                easy.append(line)
            elif d == "medium":
                medium.append(line)
            elif d == "hard":
                hard.append(line)

    def calc(items):
        success = sum(1 for x in items if x.get("passed"))
        return success, len(items), (success / len(items) if items else 0)

    e_s, e_n, e_r = calc(easy)
    m_s, m_n, m_r = calc(medium)
    h_s, h_n, h_r = calc(hard)

    def token_stats(items):
        total, count = 0, 0
        for item in items:
            u = item.get("usage", {})
            if isinstance(u, dict) and u:
                total += u.get("total_tokens", 0)
                count += 1
        return total / count if count else 0

    lines = [
        f"Easy: {e_s}/{e_n} = {e_r}",
        f"Medium: {m_s}/{m_n} = {m_r}",
        f"Hard: {h_s}/{h_n} = {h_r}",
        "",
        f"Easy Token Stats: Avg {token_stats(easy):.2f} tokens/question",
        f"Medium Token Stats: Avg {token_stats(medium):.2f} tokens/question",
        f"Hard Token Stats: Avg {token_stats(hard):.2f} tokens/question",
    ]

    content = "\n".join(lines)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\nResults saved to: {output_path}\n{content}")
    return content

# ── Main Process ──────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Text-to-Code Evaluation (no images)")
    parser.add_argument("--api_config", type=str,
                        default="src/configs/openai_api_key_config.json",
                        help="Path to API config JSON")
    parser.add_argument("--data_path", type=str,
                        default="data/MATH/MATH.jsonl",
                        help="Path to dataset JSONL")
    parser.add_argument("--output_dir", type=str,
                        default="/home/zhangxu/flowchart/CodeVision/output",
                        help="Base output directory")
    parser.add_argument("--prompt_type", type=str, default="default",
                        choices=["default", "cot"],
                        help="Prompt type: default or cot (chain-of-thought)")
    parser.add_argument("--cot", action="store_true",
                        help="Use chain-of-thought prompt (equivalent to --prompt_type cot)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing samples.jsonl (skip completed tasks)")
    args = parser.parse_args()

    print("=" * 60)
    print("Text-to-Code Evaluation (no images)")
    print("=" * 60)
    print(json.dumps(vars(args), indent=4))

    api_config = json.load(open(args.api_config))
    print("\nAPI Config:")
    print(json.dumps(api_config, indent=4))

    problems = read_jsonl_file(args.data_path)
    print(f"\nLoaded {len(problems)} problems from {args.data_path}")

    # Build output directory: output_dir/<dataset_name>/<model_name>/
    dataset_name = os.path.basename(os.path.dirname(args.data_path))
    model_name = api_config.get("model", "unknown").split("/")[-1]
    output_dir = os.path.join(args.output_dir, dataset_name, f"{model_name}-text")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    save_path = os.path.join(output_dir, "samples.jsonl")

    # ── Resume from checkpoint (optional) ──────────────────────────────────
    if args.resume:
        try:
            done_data = read_jsonl_file(save_path)
            done_ids = [x["task_id"] for x in done_data]
            problems = [x for x in problems if x["task_id"] not in done_ids]
            print(f"Resuming: skipping {len(done_ids)} done, {len(problems)} remaining")
        except Exception as e:
            print(f"Resume disabled: {e}")

    # ── Generate code ─────────────────────────────────────────────────────
    prompt_fn = construct_prompt_cot_text if (args.cot or args.prompt_type == "cot") else construct_prompt_text

    with open(save_path, "a") as f:
        for i, problem in tqdm(enumerate(problems), desc="Generating code", total=len(problems)):
            task_id = problem["task_id"]
            prompt = prompt_fn(problem)

            start_time = time.time()
            responses, usage, model_name_resp, retry_time = get_response_text(prompt, api_config)
            duration = time.time() - start_time

            response_text = responses[0]
            completion = extract_code(response_text)

            problem["response"] = response_text
            problem["completion"] = completion
            problem["usage"] = usage
            problem["model"] = str(model_name_resp)
            problem["duration"] = duration

            f.write(json.dumps(problem, ensure_ascii=False) + '\n')
            f.flush()

    # ── Token statistics ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Token Statistics")
    print("=" * 60)
    all_results = read_jsonl_file(save_path)
    token_stats = calculate_token_statistics(all_results)
    if token_stats:
        print(f"Total tasks: {token_stats['total_tasks']}")
        print(f"Total prompt tokens: {token_stats['total_prompt_tokens']}")
        print(f"Total completion tokens: {token_stats['total_completion_tokens']}")
        print(f"Total tokens: {token_stats['total_tokens']}")
        print(f"Avg prompt tokens/task: {token_stats['avg_prompt_tokens']:.2f}")
        print(f"Avg completion tokens/task: {token_stats['avg_completion_tokens']:.2f}")
        print(f"Avg total tokens/task: {token_stats['avg_total_tokens']:.2f}")

        token_stats_path = save_path.replace("samples.jsonl", "token_stats.json")
        with open(token_stats_path, "w", encoding="utf-8") as f:
            json.dump(token_stats, f, indent=4, ensure_ascii=False)
        print(f"Token stats saved to: {token_stats_path}")

    # ── Functional correctness evaluation ──────────────────────────────────
    print("\n" + "=" * 60)
    print("Running Functional Correctness Evaluation")
    print("=" * 60)
    run_evaluation(save_path, args.data_path)

    # ── Calculate and save scores ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Scoring")
    print("=" * 60)
    results_file = save_path + "_results.jsonl"
    results_path = save_path.replace("samples.jsonl", "results.txt")
    score_results(results_file, results_path)
