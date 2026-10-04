import re
import argparse
import json
import os
import fnmatch
from tqdm import tqdm
from openai import OpenAI
import anthropic
import os
import base64
import requests
from PIL import Image

# Lazy imports for local models (only needed when not using API)
# import torch
# from transformers import AutoModel, AutoTokenizer
# from transformers import MllamaForConditionalGeneration, AutoProcessor
# from transformers import AutoModelForCausalLM 


def read_jsonl_file(file_path):
    results = []
    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
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

def extract_code(content):
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
    except:
        return content

def extract_code_block_only(content):
    """Extract only the code block content; return empty string if no code block is found."""
    try:
        if '```python' in content:
            p_code = re.compile(r'```python\n(.*?)\n```', flags=re.DOTALL)
            code_blocks = p_code.findall(content)
            if code_blocks:
                code_block = code_blocks[0]
                if "assert" in code_block:
                    code_block = code_block.split("assert")[0]
                return code_block
        if '```' in content:
            p_code = re.compile(r'```(.*?)\n(.*?)```', flags=re.DOTALL)
            code_blocks = p_code.findall(content)
            if code_blocks:
                code_block = code_blocks[0][1]
                if "assert" in code_block:
                    code_block = code_block.split("assert")[0]
                return code_block
        return ""
    except Exception:
        return ""


def extract_plan(content):
    """Extract the plan text before the code block; return empty string if not found."""
    try:
        if not content:
            return ""

        code_start = content.find("```")
        plan_text = content[:code_start] if code_start != -1 else content
        plan_text = re.sub(r'^\s*Plan\s*:\s*', '', plan_text, flags=re.IGNORECASE)
        plan_text = plan_text.strip()
        return plan_text
    except Exception:
        return ""


def normalize_response_tuple(result_tuple, default_model_name=""):
    """Normalize return values from different backends, filling in retry_duration."""
    if len(result_tuple) == 4:
        responses, usage, model_name, retry_duration = result_tuple
    else:
        responses, usage, model_name = result_tuple
        retry_duration = 0
    return responses, usage, model_name or default_model_name, retry_duration


def merge_usage_dicts(*usages):
    """Merge usage from multiple calls for subsequent total token statistics."""
    merged = {}
    has_usage = False

    for usage in usages:
        if not isinstance(usage, dict):
            continue

        has_usage = True
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                merged[key] = merged.get(key, 0) + value
            elif key not in merged:
                merged[key] = value

    return merged if has_usage else None


def run_two_stage_self_planning(starter_code, image_path, api_config, response_fn, description=None):
    """First generate a plan, then generate code based on the plan. If description is provided, include the problem description in the prompt."""
    plan_prompt = construct_plan_prompt(starter_code, description)
    plan_tuple = normalize_response_tuple(
        response_fn(plan_prompt, image_path),
        default_model_name=api_config.get("model", "")
    )
    plan_responses, plan_usage, model_name, plan_retry_duration = plan_tuple
    plan_response = plan_responses[0] if plan_responses else ""
    plan_text = extract_plan(plan_response)

    if not plan_text:
        raise ValueError("self_planning first stage did not generate a valid plan")

    code_prompt = construct_code_prompt_with_plan(starter_code, plan_text, description)
    code_tuple = normalize_response_tuple(
        response_fn(code_prompt, image_path),
        default_model_name=model_name
    )
    code_responses, code_usage, code_model_name, code_retry_duration = code_tuple

    return {
        "plan": plan_text,
        "plan_response": plan_response,
        "plan_usage": plan_usage,
        "code_response": code_responses[0] if code_responses else "",
        "code_usage": code_usage,
        "usage": merge_usage_dicts(plan_usage, code_usage) or {},
        "model": code_model_name or model_name,
        "retry_duration": plan_retry_duration + code_retry_duration,
    }



#  base64 encoding format
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def _build_model_candidates(api_config):
    """Build a list of model candidate names, prioritizing the primary model name in config, then trying common aliases."""
    candidates = []
    model_name = str(api_config.get("model", "")).strip()
    if model_name:
        candidates.append(model_name)

    # Support explicitly providing aliases in config
    for alias in api_config.get("model_aliases", []):
        alias_str = str(alias).strip()
        if alias_str:
            candidates.append(alias_str)

    # Automatically add common aliases
    if "/" in model_name:
        basename = model_name.split("/")[-1].strip()
        if basename:
            candidates.append(basename)
            candidates.append(basename.lower())
        candidates.append(model_name.lower())

    # Deduplicate and maintain order
    dedup = []
    seen = set()
    for item in candidates:
        if item and item not in seen:
            dedup.append(item)
            seen.add(item)
    return dedup


def _looks_like_repetition_loop(text: str) -> bool:
    """Cheap detector for degenerate decode loops (repeated suffix / lines).

    Ported from LDB programming/generators/model.py so evaluate_all can stop
    wasting max_tokens on degenerate repetition.
    """
    if len(text) < 800:
        return False
    lines = text.splitlines()
    if len(lines) >= 8:
        last = lines[-1]
        if last.strip() and all(line == last for line in lines[-8:]):
            return True
    for n in (40, 80, 160):
        if len(text) < n * 4:
            continue
        suffix = text[-n:]
        if not suffix.strip():
            continue
        if text[-n * 4:-n].count(suffix) >= 3:
            return True
    return False


def _truncate_repetition_tail(text: str) -> str:
    """Cut the degenerate repetition tail from a completed response."""
    if not _looks_like_repetition_loop(text):
        return text
    # Binary-search the shortest prefix that still triggers the detector;
    # that prefix approximates where the degenerate loop begins.
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi) // 2
        if _looks_like_repetition_loop(text[:mid]):
            hi = mid
        else:
            lo = mid + 1
    if lo < len(text):
        print(f"[repetition] truncated degenerate tail: {len(text)} -> {lo} chars")
    return text[:lo]


def _stream_chat_completions(url, headers, payload_dict, api_config):
    """Call OpenAI-compatible API via SSE streaming, returns (responses, usage, model_name)"""
    import sys
    import time as _time

    stream_data = dict(payload_dict)
    stream_data["stream"] = True
    stream_data["stream_options"] = {"include_usage": True}

    model_name = stream_data.get("model", "unknown")
    stream_label = f"[stream][{model_name}]"

    timeout = api_config.get("timeout", 600)
    response = requests.post(url, headers=headers, json=stream_data, timeout=timeout, stream=True)

    if response.status_code != 200:
        raise Exception(f"API request failed: {response.status_code} - {response.text[:500]}")

    content_parts = []
    usage = {}
    printed_header = False
    chars_since_check = 0
    loop_abort = False

    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.strip()
        if not line or not line.startswith("data:"):
            continue

        payload = line[5:].strip()
        if payload == "[DONE]":
            break

        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue

        chunk_usage = chunk.get("usage")
        if isinstance(chunk_usage, dict):
            usage = chunk_usage

        for choice in chunk.get("choices", []):
            delta = choice.get("delta") or choice.get("message") or {}
            chunk_text = ""
            if isinstance(delta.get("content"), str):
                chunk_text = delta["content"]
            elif isinstance(delta.get("content"), list):
                for item in delta["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        chunk_text += item.get("text", "")
            rc = delta.get("reasoning_content")
            if isinstance(rc, str):
                chunk_text += rc

            if not chunk_text:
                continue

            content_parts.append(chunk_text)
            nchars = sum(len(x) for x in content_parts)
            chars_since_check += len(chunk_text)
            if chars_since_check >= 400:
                chars_since_check = 0
                if _looks_like_repetition_loop("".join(content_parts)):
                    print(f"[stream] repetition loop at {nchars} chars, stopping stream")
                    loop_abort = True
                    break

            if not printed_header:
                sys.stdout.write(f"\n{stream_label}\n")
                sys.stdout.flush()
                printed_header = True

            sys.stdout.write(chunk_text)
            sys.stdout.flush()

        if loop_abort:
            try:
                response.close()
            except Exception:
                pass
            break

    if printed_header:
        sys.stdout.write("\n")
        sys.stdout.flush()

    if isinstance(usage, dict):
        usage["max_completion_tokens"] = usage.get("completion_tokens", 0)

    full_text = _truncate_repetition_tail("".join(content_parts))
    return [full_text] if full_text else [""], usage, model_name


def get_response(prompt,image_path,api_config, max_retries=3, retry_delay=5):
    import time
    base64_image = encode_image(image_path)
    base_url = api_config["base_url"]
    api_key = api_config["api_key"]
    model_candidates = _build_model_candidates(api_config)
    if not model_candidates:
        model_candidates = [api_config.get("model", "")]

    payload_dict = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    },
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

    # Only add top_p parameter if it exists in the config file
    if "top_p" in api_config:
        payload_dict["top_p"] = api_config["top_p"]

    # Only add thinking parameter if it exists in the config file
    if "thinking" in api_config:
        payload_dict["thinking"] = api_config["thinking"]

    # Only add thinking_mode parameter if it exists in the config file (for intern-s1/intern-s1-mini deep thinking mode)
    if "thinking_mode" in api_config:
        payload_dict["thinking_mode"] = api_config["thinking_mode"]

    # Support directly providing the full API endpoint path, e.g. .../v1/messages
    # Such URLs no longer auto-append /chat/completions.
    base_url = base_url.rstrip('/')

    if api_config.get("use_direct_messages_url", False) or base_url.endswith("/v1/messages"):
        url = base_url
    elif base_url.endswith("/v1"):
        url = base_url + "/chat/completions"
    elif base_url.endswith("/v4"):
        url = base_url + "/chat/completions"
    else:
        url = base_url + "/v1/chat/completions"

    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }

    # 429 rate limit error counter
    rate_limit_retries = 0
    max_rate_limit_retries = 3
    total_retry_time = 0

    # Streaming mode: use SSE directly, no model alias retry
    if api_config.get("stream", False):
        payload_dict["model"] = model_candidates[0]
        return _stream_chat_completions(url, headers, payload_dict, api_config) + (total_retry_time,)

    for model_idx, model_candidate in enumerate(model_candidates):
        if model_idx > 0:
            print(f"Trying model alias: {model_candidate}")
        payload_dict["model"] = model_candidate
        payload = json.dumps(payload_dict, ensure_ascii=False)

        switch_to_next_model = False
        for attempt in range(max_retries):
            try:
                # Set a longer timeout (10 minutes) as some models may need more time to process images
                timeout = api_config.get("timeout", 600)
                response = requests.post(url, headers=headers, json=payload_dict, timeout=timeout)

                # Check response status code
                if response.status_code != 200:
                    error_text = response.text[:500]

                    # Special handling for 429 rate limit errors
                    if response.status_code == 429:
                        rate_limit_retries += 1
                        if rate_limit_retries <= max_rate_limit_retries:
                            print(f"Encountered 429 rate limit error (attempt {rate_limit_retries}/{max_rate_limit_retries})")
                            print(f"Waiting 30 seconds before retry...")
                            time.sleep(30)
                            total_retry_time += 30
                            continue  # Continue to next attempt
                        else:
                            print(f"429 rate limit error reached max retries ({max_rate_limit_retries}), giving up")
                            raise Exception(f"API request failed: 429 Rate Limit Exceeded (retried {max_rate_limit_retries} times)")

                    # Common local service error: current model name/routing is incompatible with service implementation, try model aliases
                    has_next_model = model_idx < len(model_candidates) - 1
                    lower_error = error_text.lower()
                    if has_next_model and response.status_code in (400, 404) and ("model" in lower_error and ("not found" in lower_error or "does not exist" in lower_error)):
                        print(f"Model name {model_candidate} may not be registered on the server, trying next alias...")
                        switch_to_next_model = True
                        break

                    if response.status_code == 500 and "has no attribute 'generate'" in lower_error:
                        if has_next_model:
                            print(f"Model name {model_candidate} is unavailable in the current local service (server reports missing generate), trying next alias...")
                            switch_to_next_model = True
                            break
                        # All model candidates hit the same server error, this is an unrecoverable error, fail directly
                        raise RuntimeError(
                            "UNRECOVERABLE_BACKEND_ERROR: Local service returned "
                            "'Glm4vModel' object has no attribute 'generate', "
                            "indicating the current backend implementation does not support this model's chat/completions inference."
                        )

                    print(f"API request failed: {response.status_code}")
                    safe_error = error_text.encode('ascii', 'replace').decode('ascii')
                    print(f"Response content: {safe_error}")
                    raise Exception(f"API request failed: {response.status_code}")

                # Check if response content is empty
                if not response.text.strip():
                    raise Exception("API returned empty response")

                # Try to parse JSON
                try:
                    response_json = response.json()
                except json.JSONDecodeError as e:
                    safe_error = str(e).encode('ascii', 'replace').decode('ascii')
                    print(f"JSON parsing failed: {safe_error}")
                    print(f"Response status code: {response.status_code}")
                    safe_headers = str(dict(response.headers)).encode('ascii', 'replace').decode('ascii')
                    print(f"Response headers: {safe_headers}")
                    print(f"Content-Type: {response.headers.get('Content-Type', 'not set')}")
                    safe_text = response.text[:2000].encode('ascii', 'replace').decode('ascii')
                    print(f"Response content (first 2000 chars): {safe_text}")
                    raise Exception(f"API returned invalid JSON: {e}")

                # Extract response content, prefer content field over reasoning_content
                responses = []
                for i in range(api_config["n"]):
                    message = response_json["choices"][i]["message"]
                    # Ensure we use the content field, not reasoning_content
                    content = message.get("content", "")

                    # If content is empty or looks incomplete, check if it was truncated
                    if not content or not content.strip():
                        print(f"Warning: content of response {i} is empty")
                        # Check if reasoning_content exists
                        if "reasoning_content" in message:
                            print(f"Warning: reasoning_content detected, but should not be used as primary output")

                    responses.append(_truncate_repetition_tail(content))

                # Add max_completion_tokens info in usage (record actual max completion_tokens)
                usage = response_json.get("usage", {})
                if isinstance(usage, dict):
                    usage["max_completion_tokens"] = usage.get("completion_tokens", 0)

                # Check if truncated due to length limit
                for i, choice in enumerate(response_json.get("choices", [])):
                    if choice.get("finish_reason") == "length":
                        print(f"Warning: response {i} was truncated due to max_tokens limit")
                        print(f"Current max_tokens: {api_config.get('max_tokens', 'unknown')}")

                return responses, usage, response_json.get("model", model_candidate), total_retry_time
            except Exception as e:
                # If 429 error and max retries not reached, already handled above, skip here
                if "429" in str(e) and rate_limit_retries <= max_rate_limit_retries:
                    continue

                # Unrecoverable errors should not be retried, raise directly to stop current dataset
                if isinstance(e, RuntimeError) and "UNRECOVERABLE_BACKEND_ERROR" in str(e):
                    raise

                if switch_to_next_model:
                    break

                safe_e = str(e).encode('ascii', 'replace').decode('ascii')
                print(f"Attempt {attempt + 1}/{max_retries} failed for task: {safe_e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                    total_retry_time += retry_delay
                else:
                    print(f"Model {model_candidate} has reached max retries")
                    # Current model name exhausted, try next model alias
                    break

    print(f"All model candidates exhausted. Skipping this task.")
    # Return empty response to allow evaluation to continue
    return ["ERROR: API request failed"], None, api_config.get("model", ""), total_retry_time

def get_response_claude(prompt,image_path,api_config):
    base64_image = encode_image(image_path)
    Baseurl = api_config["base_url"]
    Skey = api_config["api_key"]
    payload_dict = {
        "model": api_config["model"],
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
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
    # Only add top_p parameter if it exists in the config file
    if "top_p" in api_config:
        payload_dict["top_p"] = api_config["top_p"]

    payload = json.dumps(payload_dict)
    # Check if base_url already contains /v1, avoid duplication
    if Baseurl.endswith("/v1"):
        url = Baseurl + "/chat/completions"
    else:
        url = Baseurl + "/v1/chat/completions"
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {Skey}',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, json=payload_dict)
    response = response.json()
    # print(response)
    responses = [response["choices"][i]["message"]["content"] for i in range(api_config["n"])]
    return responses, response["usage"], response["model"]

def get_response_internvl(prompt,image_path,api_config):
    import requests

    # Check if using Anthropic-compatible API (chat.intern-ai.org.cn)
    base_url = api_config["base_url"]
    if "chat.intern-ai.org.cn" in base_url:
        # Use Anthropic-compatible API
        base64_image = encode_image(image_path)
        url = base_url.rstrip('/') + "/v1/messages"

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_config["api_key"],
            "anthropic-version": "2023-06-01"
        }

        # Build Anthropic format message
        messages = []
        # Add image
        with open(image_path, 'rb') as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')

        content = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": image_data
                }
            },
            {
                "type": "text",
                "text": prompt
            }
        ]

        data = {
            "model": api_config["model"],
            "max_tokens": api_config.get("max_tokens", 1024),
            "messages": [
                {
                    "role": "user",
                    "content": content
                }
            ]
        }

        # Add optional parameters
        if "thinking_mode" in api_config:
            data["thinking_mode"] = api_config["thinking_mode"]

        try:
            response = requests.post(url, headers=headers, json=data, timeout=600)

            if response.status_code != 200:
                print(f"Anthropic API request failed: {response.status_code}")
                safe_text = response.text[:500].encode('ascii', 'replace').decode('ascii')
                print(f"Response content: {safe_text}")
                raise Exception(f"Anthropic API request failed: {response.status_code}")

            result = response.json()
            # Anthropic format: content[0].text
            reply = result["content"][0]["text"]
            responses = [reply]

            # Anthropic format usage
            usage = result.get("usage", {})

            return responses, usage, api_config["model"]

        except Exception as e:
            safe_e = str(e).encode('ascii', 'replace').decode('ascii')
            print(f"Anthropic API call exception: {safe_e}")
            return [f"ERROR: {safe_e}"], None, api_config["model"]

    # Check if using OpenAI-compatible API (base_url ends with /v1 or /v4)
    if base_url.rstrip('/').endswith('/v1') or base_url.rstrip('/').endswith('/v4'):
        # Use OpenAI-compatible API
        return get_response(prompt, image_path, api_config)[:3]

    # Use legacy ModelScope API
    url = api_config["base_url"]  # （API）
    api_key = api_config["api_key"]  # （KEY）

    # example
    file_paths = [
        image_path
    ]
    question = prompt  # (Question)

    files = [('files', open(file_path, 'rb')) for file_path in file_paths]
    data = {
        'question': question,
        'api_key': api_key
    }

    responses = []
    try:
        response = requests.post(url, files=files, data=data)
        responses.append(response.json().get("response", "No response key found in the JSON."))
        if response.status_code == 200:
            pass
            # print("Response:", response.json().get("response", "No response key found in the JSON."))
        else:
            safe_text = str(response.text).encode('ascii', 'replace').decode('ascii')
            print("Error:", response.status_code, safe_text)
    except requests.exceptions.RequestException as e:
        safe_error = str(e).encode('ascii', 'replace').decode('ascii')
        print(f"Error: {safe_error}")
    return responses, response.json().get("usage", 0), response.json().get("model", "InternVL-2-Pro")

def get_response_minicpm(prompt,image_path,api_config,model,tokenizer):
    image = Image.open(image_path).convert('RGB')
    question = prompt
    msgs = [{'role': 'user', 'content': [image, question]}]

    res = model.chat(
        image=None,
        msgs=msgs,
        tokenizer=tokenizer
    )
    responses = [res]
    return responses, 0, api_config["model"]

def get_response_phi(prompt,image_path,api_config,model,processor):
    messages = [ 
        {"role": "user", "content": f"<|image_1|>\n{prompt}"}
    ] 
    image = Image.open(image_path)
    prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(prompt, [image], return_tensors="pt").to(model.device)
    generation_args = { 
        "max_new_tokens": api_config["max_tokens"], 
        "temperature": api_config["temperature"], 
        "do_sample": True, 
        "top_p":api_config["top_p"],
    } 
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args) 
    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    responses = [response]
    return responses, 0, api_config["model"]

def get_response_llama(prompt,image_path,api_config,model,processor):
    image = Image.open(image_path)
    messages = [
        {"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": prompt}
        ]}
    ]
    input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(
        image,
        input_text,
        add_special_tokens=False,
        return_tensors="pt"
    ).to(model.device)
    generation_args = { 
        "max_new_tokens": api_config["max_tokens"], 
        "temperature": api_config["temperature"], 
        "do_sample": True, 
        "top_p":api_config["top_p"],
    } 
    generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id, **generation_args)
    # remove input tokens 
    generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
    response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    responses = [response]
    return responses, 0, api_config["model"]



def _is_stdin_starter(starter_code, description=None):
    """Detect stdin/stdout (AtCoder-style) tasks from starter code or description."""
    starter = starter_code or ""
    text = description or ""
    return (
        ("def solve(" in starter and "class Solution" not in starter)
        or "Standard Input" in text
        or "standard input" in text
    )


def _io_prompt_note(is_stdin):
    """Build an IO note matching the task contract."""
    if is_stdin:
        return (
            "Note: Read input from stdin using input() or sys.stdin.readline(), "
            "and print the answer using print(). The solve() function will be "
            "called automatically; do not call it yourself."
        )
    return (
        "Note: If you're using a specific python package, you'll need to import it "
        "yourself. Do not use input() to read input; just complete the python function."
    )


def construct_prompt(starter_code):
    is_stdin = _is_stdin_starter(starter_code)
    prompt = f"""Generate code according to flowchart.
{_io_prompt_note(is_stdin)}
Starter Code:
___BT___python
%%%starter_code%%%
___BT___
Present the code between ___BT___python and ___BT___.
"""
    return prompt.replace("%%%starter_code%%%", starter_code).replace("___BT___", chr(96) * 3)


def construct_prompt_with_description(starter_code, description):
    """Build prompt with problem description, for Zero-Shot-CoT and self_planning modes."""
    is_stdin = _is_stdin_starter(starter_code, description)
    prompt = f"""Generate code according to flowchart and problem description.

Problem Description:
%%%description%%%

Starter Code:
___BT___python
%%%starter_code%%%
___BT___
{_io_prompt_note(is_stdin)}
Present the code between ___BT___python and ___BT___.
"""
    return prompt.replace("%%%starter_code%%%", starter_code).replace("%%%description%%%", description).replace("___BT___", chr(96) * 3)


def construct_prompt_cot(starter_code, description=None):
    """Build CoT prompt. If description is provided, include the problem description."""
    io_note = _io_prompt_note(_is_stdin_starter(starter_code, description))
    if description:
        prompt = f"""Generate code according to flowchart and problem description.
Before writing the code, think step by step about the control flow and variables.

Problem Description:
%%%description%%%

Rules:
- Put your reasoning outside the code block.
- The code block must contain only Python code.

Starter Code:
___BT___python
%%%starter_code%%%
___BT___
{io_note}
Present the code between ___BT___python and ___BT___.
"""
        return prompt.replace("%%%starter_code%%%", starter_code).replace("%%%description%%%", description).replace("___BT___", chr(96) * 3)
    else:
        prompt = f"""Generate code according to flowchart.
Before writing the code, think step by step about the control flow and variables.

Rules:
- Put your reasoning outside the code block.
- The code block must contain only Python code.

Starter Code:
___BT___python
%%%starter_code%%%
___BT___
{io_note}
Present the code between ___BT___python and ___BT___.
"""
        return prompt.replace("%%%starter_code%%%", starter_code).replace("___BT___", chr(96) * 3)


def construct_plan_prompt(starter_code, description=None):
    """Build planning prompt. If description is provided, include the problem description."""
    if description:
        prompt = """You are given a flowchart image and a starter code signature.
Analyze the flowchart and problem description, then produce a concise implementation plan.

Problem Description:
%%%description%%%

Focus on the control flow, conditions, loops, and key variables.
Do not write code in this step.

Starter Code:
```python
%%%starter_code%%%
```

Output Format:
Plan:
1. ...
2. ...
"""
        return prompt.replace("%%%starter_code%%%", starter_code).replace("%%%description%%%", description)
    else:
        prompt = """You are given a flowchart image and a starter code signature.
Analyze the flowchart and produce a concise implementation plan.

Focus on the control flow, conditions, loops, and key variables.
Do not write code in this step.

Starter Code:
```python
%%%starter_code%%%
```

Output Format:
Plan:
1. ...
2. ...
"""
        return prompt.replace("%%%starter_code%%%", starter_code)


def construct_code_prompt_with_plan(starter_code, plan_text, description=None):
    """Build prompt for code generation based on plan. If description is provided, include the problem description."""
    # LCB AtCoder tasks use standard input; other tasks keep the existing functional prompt.
    is_stdin = _is_stdin_starter(starter_code, description)
    io_instruction = (
        "- Read input from stdin using input() or sys.stdin.readline().\n"
        "- Print the answer using print().\n"
        "- The solve() function will be called automatically; do not call it yourself."
        if is_stdin else
        "- Do not use input(); just complete the target function."
    )
    if description:
        prompt = """You are given a flowchart image, a starter code signature, and a plan produced in a previous step.
Generate the final Python code according to the flowchart and follow the plan closely.

Problem Description:
%%%description%%%

Rules:
- Output only one Python code block.
- Do not repeat the plan.
- Do not add any explanation outside the code block.
- If you need a specific python package, import it yourself.
%%%io_instruction%%%

Starter Code:
```python
%%%starter_code%%%
```

Plan:
%%%plan_text%%%

Present the final code between ```python and ```.
"""
        return (
            prompt.replace("%%%starter_code%%%", starter_code)
            .replace("%%%plan_text%%%", plan_text)
            .replace("%%%description%%%", description)
            .replace("%%%io_instruction%%%", io_instruction)
        )
    else:
        prompt = """You are given a flowchart image, a starter code signature, and a plan produced in a previous step.
Generate the final Python code according to the flowchart and follow the plan closely.

Rules:
- Output only one Python code block.
- Do not repeat the plan.
- Do not add any explanation outside the code block.
- If you need a specific python package, import it yourself.
%%%io_instruction%%%

Starter Code:
```python
%%%starter_code%%%
```

Plan:
%%%plan_text%%%

Present the final code between ```python and ```.
"""
        return (
            prompt.replace("%%%starter_code%%%", starter_code)
            .replace("%%%plan_text%%%", plan_text)
            .replace("%%%io_instruction%%%", io_instruction)
        )

def construct_prompt_with_plan(starter_code):
    """Backward-compatible alias, kept as the first stage plan prompt."""
    return construct_plan_prompt(starter_code)

def construct_prompt_mask(starter_code):
    prompt = """Generate code according to flowchart.
Note: If you're using a specific python package, you'll need to import it yourself. You don't need to use functions like input() to get input, just complete the python function.
Note: A small part of the flowchart may be masked, you need to understand the flowchart and then generate the complete code.
Starter Code:
```python
%%%starter_code%%%
```
Present the code between ```python and ```.
"""
    return prompt.replace("%%%starter_code%%%", starter_code)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_config", type=str, default="src/configs/openai_api_key_config.json")
    parser.add_argument("--data_path", type=str, default="data/HumanEval-V/HumanEval.jsonl")
    parser.add_argument("--image_dir", type=str, default="data/HumanEval-V/images")
    parser.add_argument("--output_dir", type=str, default="/output")
    parser.add_argument("--prompt_variant", type=str, default="default", choices=["default", "self_planning", "zero_shot_cot"])
    args = parser.parse_args()
    print("-" * 20, "Args", "-" * 20)
    print(json.dumps(vars(args), indent=4))

    api_config = json.load(open(args.api_config))
    print("-" * 20, "API Config", "-" * 20)
    print(json.dumps(api_config, indent=4))

    problems = read_jsonl_file(args.data_path)
    print("Loading data from {}".format(args.data_path)," total tasks:", len(problems))
    ori_len = len(problems)
    samples = []

    dataset = args.data_path.split("/")[-2]
    output_suffix = ""
    if args.prompt_variant == "self_planning":
        output_suffix = "_self_planning"
    elif args.prompt_variant == "zero_shot_cot":
        output_suffix = "_zero_shot_cot"
    args.output_dir = os.path.join(args.output_dir,dataset, api_config["model"] + output_suffix)
    os.makedirs(args.output_dir, exist_ok=True)
    print("Update output dir to {}".format(args.output_dir))

    print("-" * 20, "Check existing results", "-" * 20)
    save_path = os.path.join(args.output_dir, "samples.jsonl")
    try:
        done_data = read_jsonl_file(save_path)
        done_ids = [x["task_id"] for x in done_data]
        print("Skipping {} tasks".format(len(done_ids)))
        problems = [x for x in problems if x["task_id"] not in done_ids]
        print("Remaining {} tasks".format(len(problems)))
        assert len(problems) + len(done_ids) == ori_len, "Skipping data and remaining tasks do not match the original data size"
    except Exception as e:
        safe_e = str(e).encode('ascii', 'replace').decode('ascii')
        print(safe_e)
        print("No existing results!")
    print("-" * 20, "Starting", "-" * 20)


    if "minicpm" in args.api_config:
        # minicpm
        model_path = api_config["api_key"]
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
            attn_implementation='sdpa', torch_dtype=torch.bfloat16) # sdpa or flash_attention_2, no eager
        model = model.eval().cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    elif "llama" in args.api_config and "api_key_config" not in args.api_config:
        model_id = api_config["api_key"]
        model = MllamaForConditionalGeneration.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        processor = AutoProcessor.from_pretrained(model_id)
    elif "phi_" in args.api_config:
        model_id = api_config["api_key"]
        model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='eager') # use _attn_implementation='eager' to disable flash attention
        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 

    def response_fn(prompt, image_path):
        if "internvl" in args.api_config:
            return get_response_internvl(prompt, image_path, api_config)
        if "claude" in args.api_config:
            return get_response_claude(prompt, image_path, api_config)
        if "minicpm" in args.api_config:
            return get_response_minicpm(prompt, image_path, api_config, model, tokenizer)
        if "llama" in args.api_config and "api_key_config" not in args.api_config:
            return get_response_llama(prompt, image_path, api_config, model, processor)
        if "phi_" in args.api_config:
            return get_response_phi(prompt, image_path, api_config, model, processor)
        return get_response(prompt, image_path, api_config)

    with open(save_path, "a") as f:
        for i, problem in tqdm(enumerate(problems),desc="Generating samples",total=len(problems)):
            task_id = problem["task_id"]
            image_path = os.path.join(args.image_dir, task_id + ".png")
            # Get problem description
            problem_description = problem.get("prompt", "").strip()

            if "MASK" in args.image_dir:
                prompt = construct_prompt_mask(problem["starter_code"])
                responses, usage, model_name, _ = normalize_response_tuple(
                    response_fn(prompt, image_path),
                    default_model_name=api_config.get("model", "")
                )
                problem["response"] = responses[0]
                problem["completion"] = extract_code(responses[0])
                problem["usage"] = usage
                problem["model"] = str(model_name)
            elif args.prompt_variant == "self_planning":
                # self_planning mode: pass problem description
                planning_result = run_two_stage_self_planning(
                    problem["starter_code"],
                    image_path,
                    api_config,
                    response_fn,
                    description=problem_description,  # Pass problem description
                )
                problem["response"] = planning_result["code_response"]
                problem["completion"] = extract_code(planning_result["code_response"])
                problem["usage"] = planning_result["usage"]
                problem["plan"] = planning_result["plan"]
                problem["plan_response"] = planning_result["plan_response"]
                if planning_result["plan_usage"] is not None:
                    problem["plan_usage"] = planning_result["plan_usage"]
                if planning_result["code_usage"] is not None:
                    problem["code_usage"] = planning_result["code_usage"]
                problem["model"] = str(planning_result["model"])
            else:
                if args.prompt_variant == "zero_shot_cot":
                    # Zero-Shot-CoT mode: pass problem description
                    prompt = construct_prompt_cot(problem["starter_code"], problem_description)
                else:
                    prompt = construct_prompt(problem["starter_code"])

                responses, usage, model_name, _ = normalize_response_tuple(
                    response_fn(prompt, image_path),
                    default_model_name=api_config.get("model", "")
                )
                problem["response"] = responses[0]
                problem["completion"] = extract_code(responses[0])
                problem["usage"] = usage
                problem["model"] = str(model_name)

            f.write(
                json.dumps(problem) + "\n"
            )
            f.flush()  # make sure the output is written to file

    # Calculate token statistics
    print("-" * 20, "Token Statistics", "-" * 20)
    all_results = read_jsonl_file(save_path)
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    valid_count = 0

    for result in all_results:
        usage = result.get("usage", {})
        if isinstance(usage, dict) and usage:
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens_usage = usage.get("total_tokens", 0)

            total_prompt_tokens += prompt_tokens
            total_completion_tokens += completion_tokens
            total_tokens += total_tokens_usage
            valid_count += 1

    if valid_count > 0:
        avg_prompt_tokens = total_prompt_tokens / valid_count
        avg_completion_tokens = total_completion_tokens / valid_count
        avg_total_tokens = total_tokens / valid_count

        print(f"Total tasks: {valid_count}")
        print(f"Total prompt tokens: {total_prompt_tokens}")
        print(f"Total completion tokens: {total_completion_tokens}")
        print(f"Total tokens: {total_tokens}")
        print(f"Average prompt tokens per task: {avg_prompt_tokens:.2f}")
        print(f"Average completion tokens per task: {avg_completion_tokens:.2f}")
        print(f"Average total tokens per task: {avg_total_tokens:.2f}")

        # Save token statistics to file
        token_stats = {
            "total_tasks": valid_count,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "avg_prompt_tokens": avg_prompt_tokens,
            "avg_completion_tokens": avg_completion_tokens,
            "avg_total_tokens": avg_total_tokens
        }
        token_stats_path = save_path.replace("samples.jsonl", "token_stats.json")
        with open(token_stats_path, "w", encoding="utf-8") as f:
            json.dump(token_stats, f, indent=4, ensure_ascii=False)
        print(f"Token statistics saved to: {token_stats_path}")
    else:
        print("No valid usage data found for token statistics.")

    # execute bash
    cmd = f"evaluate_functional_correctness {save_path} --problem_file={args.data_path}"
    os.system(cmd)


    # calculate score
    results_file = save_path + "_results.jsonl"
    results = read_jsonl_file(results_file)
    from score_baseline import Score
    results_fine_grained = Score(results)
    print(results_fine_grained)
    result_save_path = save_path.replace("samples.jsonl", "results.txt")
    # write results as string to file
    with open(result_save_path, "a") as f:
        f.write(str(results_fine_grained))

    
