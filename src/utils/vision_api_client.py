"""
Vision model API client
Used to read config and call vision model API
"""

import json
import base64
import re
import requests
import time
import sys
from typing import Dict, Any, Optional
from pathlib import Path


class VisionAPIClient:
    """Vision model API client"""
    
    def __init__(self, config_dict: Dict[str, Any] = None, config_path: str = None):
        """
        Initialize the vision model API client
        
        Args:
            config_dict: Directly provide a config dictionary (recommended), format consistent with src
            config_path: Config file path, if provided the file will be loaded
        """
        if config_dict is not None:
            self.config = config_dict
        elif config_path is not None:
            self.config = self._load_config(config_path)
        else:
            raise ValueError("Must provide either config_dict or config_path parameter")
        
        # Create a session with retry mechanism
        self.session = self._create_session_with_retry()
        self.request_counter = 0
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load config file"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Config file format error: {e}")    
    def _create_session_with_retry(self):
        """Create a session (retry logic handled by custom methods)"""
        session = requests.Session()
        # Do not use Retry, as retry logic is handled in _call_api_with_retry and _call_api_with_retry_text
        return session

    def _get_request_timeout(self, config: Dict[str, Any]) -> Any:
        """
        Get requests timeout configuration.
        Prefer (connect_timeout, read_timeout); fall back to timeout if not configured.
        """
        connect_timeout = config.get("connect_timeout", 20)
        read_timeout = config.get("read_timeout", config.get("timeout", 1200))
        return (connect_timeout, read_timeout)

    def _is_retryable_status(self, status_code: int) -> bool:
        """Determine whether an HTTP status code is retryable"""
        retryable_status_codes = {
            408, 409, 425, 429,
            500, 502, 503, 504,
            520, 521, 522, 523, 524
        }
        return status_code in retryable_status_codes

    def _get_retry_wait_time(self, attempt: int, base_wait_time: int, max_wait_time: int) -> int:
        """Exponential backoff wait time"""
        return min(base_wait_time * (2 ** attempt), max_wait_time)

    def _safe_error_text(self, response: requests.Response, max_len: int = 300) -> str:
        """Compress error text to avoid printing large blocks of HTML that bloat the logs"""
        text = (response.text or "").strip().replace("\n", " ")
        return text[:max_len]

    def _cap_max_tokens_for_context(self, data: Dict[str, Any], response: requests.Response) -> bool:
        """Shrink max_tokens when the server rejects it as larger than the
        remaining context window.

        Local vLLM servers return 400 like: "'max_tokens' ... is too large:
        32768. This model's maximum context length is 32768 tokens and your
        request has 2034 input tokens (32768 > 32768 - 2034)."  Parse the
        reported numbers, cap max_tokens to what actually fits, and let the
        caller retry. Returns True when max_tokens was reduced.
        """
        if response.status_code != 400 or "max_tokens" not in (response.text or ""):
            return False
        match_ctx = re.search(r"maximum context length is (\d+)", response.text)
        match_input = re.search(r"request has (\d+) input tokens", response.text)
        if not match_ctx or not match_input:
            return False
        context_len = int(match_ctx.group(1))
        input_tokens = int(match_input.group(1))
        key = "max_completion_tokens" if "max_completion_tokens" in data else "max_tokens"
        current = data.get(key)
        if not isinstance(current, int):
            return False
        # Safety margin for token-counting differences between server and client
        new_max = max(context_len - input_tokens - 512, 1024)
        if new_max >= current:
            return False
        print(f"[VisionAPIClient] context limit: capping {key} {current} -> {new_max} "
              f"(context length {context_len}, input tokens {input_tokens}), retrying")
        data[key] = new_max
        return True

    def _should_stream(self, config: Dict[str, Any]) -> bool:
        """Determine whether streaming output is enabled in the current config"""
        return bool(config.get("stream", False))

    def _get_stream_label(self, config: Dict[str, Any]) -> str:
        """Get streaming output label"""
        return config.get("stream_label") or config.get("model", "unknown-model")

    def _extract_text_content(self, content: Any) -> str:
        """Extract text content from OpenAI-compatible response structure"""
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text":
                        text_value = item.get("text", "")
                        if isinstance(text_value, str):
                            parts.append(text_value)
                    elif isinstance(item.get("text"), str):
                        parts.append(item["text"])
            return "".join(parts)
        if isinstance(content, dict):
            if isinstance(content.get("text"), str):
                return content["text"]
        return ""

    def _extract_stream_chunk_text(self, choice: Dict[str, Any]) -> str:
        """Extract new text from a streaming chunk"""
        delta = choice.get("delta") or choice.get("message") or {}
        text_parts = []

        content_text = self._extract_text_content(delta.get("content"))
        if content_text:
            text_parts.append(content_text)

        reasoning_content = delta.get("reasoning_content")
        if isinstance(reasoning_content, str):
            text_parts.append(reasoning_content)
        elif isinstance(reasoning_content, list):
            text_parts.append(self._extract_text_content(reasoning_content))

        return "".join(text_parts)

    def _stream_chat_completion(self, data: Dict[str, Any], headers: Dict[str, str],
                                timeout: Any, config: Dict[str, Any]) -> tuple:
        """Call OpenAI-compatible API via SSE streaming, return (result, usage)"""
        max_retries = config.get("max_retries", 8)
        base_wait_time = config.get("base_wait_time", 20)
        max_wait_time = config.get("max_wait_time", 120)
        last_error = None
        total_api_latency = 0
        total_wait_time = 0

        stream_data = dict(data)
        stream_data["stream"] = True

        self.request_counter += 1
        stream_label = self._get_stream_label(config)
        stream_prefix = f"[stream][{stream_label}][#{self.request_counter}]"

        for attempt in range(max_retries):
            time.sleep(2)
            total_wait_time += 2

            try:
                req_start_time = time.time()
                response = self.session.post(
                    f"{config.get('base_url', '').rstrip('/')}/chat/completions",
                    headers=headers,
                    json=stream_data,
                    timeout=timeout,
                    stream=True
                )
                req_end_time = time.time()
                total_api_latency += (req_end_time - req_start_time)

                if response.status_code == 200:
                    content_parts = []
                    usage = {}
                    printed_header = False
                    printed_text = False

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
                        except json.JSONDecodeError as e:
                            raise Exception(f"Streaming response JSON parse failed: {e}; raw content: {payload[:200]}")

                        chunk_usage = chunk.get("usage")
                        if isinstance(chunk_usage, dict):
                            usage = chunk_usage

                        for choice in chunk.get("choices", []):
                            chunk_text = self._extract_stream_chunk_text(choice)
                            if not chunk_text:
                                continue

                            if not printed_header:
                                sys.stdout.write(f"\n{stream_prefix}\n")
                                sys.stdout.flush()
                                printed_header = True

                            sys.stdout.write(chunk_text)
                            sys.stdout.flush()
                            printed_text = True
                            content_parts.append(chunk_text)

                    if printed_text:
                        sys.stdout.write("\n")
                        sys.stdout.flush()

                    if isinstance(usage, dict):
                        usage["max_completion_tokens"] = usage.get("completion_tokens", 0)
                        usage["api_latency"] = total_api_latency
                        usage["total_wait_time"] = total_wait_time

                    return "".join(content_parts), usage

                # vLLM/local servers reject max_tokens larger than the remaining
                # context window with a 400; shrink it to fit and retry.
                if self._cap_max_tokens_for_context(data, response):
                    continue

                if self._is_retryable_status(response.status_code):
                    wait_time = self._get_retry_wait_time(attempt, base_wait_time, max_wait_time)
                    error_brief = self._safe_error_text(response)
                    last_error = f"API request failed: {response.status_code} - {error_brief}"
                    reason = "429 rate limit error" if response.status_code == 429 else f"HTTP {response.status_code} error"
                    if attempt < max_retries - 1:
                        print(f"Encountered {reason}, waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        total_wait_time += wait_time
                        continue
                    raise Exception(f"API request failed: reached max retries {max_retries}, last error: {last_error}")

                raise Exception(f"API request failed: {response.status_code} - {self._safe_error_text(response)}")

            except requests.exceptions.RequestException as e:
                req_end_time = time.time()
                try:
                    total_api_latency += (req_end_time - req_start_time)
                except UnboundLocalError:
                    pass

                error_type = type(e).__name__
                last_error = f"Network request failed ({error_type}): {e}"
                wait_time = self._get_retry_wait_time(attempt, base_wait_time, max_wait_time)

                if attempt < max_retries - 1:
                    print(f"Network request failed ({error_type}), waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    total_wait_time += wait_time
                    continue
                raise Exception(f"Network request failed: reached max retries {max_retries} - {e}")

        raise Exception(last_error)
    
    def analyze_image(self, image_data: bytes, prompt: str) -> tuple:
        """
        Analyze image content
        
        Args:
            image_data: Image binary data
            prompt: Analysis prompt
            
        Returns:
            (Analysis result text, token usage info)
        """
        result, usage = self._call_api_with_retry(image_data, prompt)
        return result, usage
    
    def generate_text(self, prompt: str) -> tuple:
        """
        Generate text

        Args:
            prompt: Text generation prompt

        Returns:
            (Generated text, token usage info)
        """
        result, usage = self._call_api_with_retry_text(prompt)
        return result, usage

    def call_api(self, prompt: str, image_path: Optional[str] = None, max_tokens: Optional[int] = None) -> Dict[str, Any]:
        """
        Unified API call interface

        Args:
            prompt: Prompt
            image_path: Image path (optional, if provided calls vision API)
            max_tokens: Maximum token count (optional, overrides config value)

        Returns:
            Dictionary containing content and usage
        """
        # Temporarily modify max_tokens
        original_max_tokens = None
        if max_tokens is not None:
            original_max_tokens = self.config.get("max_tokens")
            self.config["max_tokens"] = max_tokens

        try:
            if image_path:
                # Call vision API
                with open(image_path, 'rb') as f:
                    image_data = f.read()
                content, usage = self.analyze_image(image_data, prompt)
            else:
                # Call text API
                content, usage = self.generate_text(prompt)

            return {
                "content": content,
                "usage": usage
            }
        finally:
            # Restore original max_tokens
            if original_max_tokens is not None:
                self.config["max_tokens"] = original_max_tokens
    
    def _call_api_with_retry(self, image_data: bytes, prompt: str, config: Dict[str, Any] = None) -> tuple:
        """Call API with retry, return (result, usage)"""
        if config is None:
            config = self.config

        # Convert image to base64
        image_base64 = base64.b64encode(image_data).decode('utf-8')

        # Build request data
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.get('api_key')}"
        }

        data = {
            "model": config.get("model"),
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
                                "url": f"data:image/png;base64,{image_base64}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": config.get("max_tokens", 2000),
            "temperature": config.get("temperature", 0.1),
            "top_p": config.get("top_p", 0.95),
            "n": config.get("n", 1),
        }

        # Only send stop when explicitly provided as non-empty in config; some models (e.g. gpt-5-mini) do not support this parameter
        stop_value = config.get("stop")
        if stop_value:
            data["stop"] = stop_value

        # Only add thinking parameter when it exists in the config file
        if "thinking" in config:
            data["thinking"] = config["thinking"]

        # Optional sampling knobs (e.g. repetition_penalty for local vLLM
        # servers where small models degenerate into repetition loops);
        # forwarded only when explicitly present in the config.
        for opt_key in ("repetition_penalty", "frequency_penalty",
                        "presence_penalty", "top_k", "min_p"):
            if config.get(opt_key) is not None:
                data[opt_key] = config[opt_key]

        # Timeout: separate connect and read; faster retry on connect failure
        timeout = self._get_request_timeout(config)

        if self._should_stream(config):
            return self._stream_chat_completion(data, headers, timeout, config)

        # Increase retry count and exponential backoff
        max_retries = config.get("max_retries", 8)
        base_wait_time = config.get("base_wait_time", 20)
        max_wait_time = config.get("max_wait_time", 120)
        last_error = None
        total_wait_time = 0  # Track retry wait time

        for attempt in range(max_retries):
            # Enforce request interval to avoid burst high-frequency calls
            time.sleep(2)
            total_wait_time += 2

            # Send request
            try:
                response = self.session.post(
                    f"{config.get('base_url', '').rstrip('/')}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=timeout
                )

                if response.status_code == 200:
                    # Check if response content is empty
                    if not response.text.strip():
                        raise Exception("API returned empty response")

                    # Try to parse JSON
                    try:
                        result = response.json()
                    except json.JSONDecodeError as e:
                        print(f"JSON parse failed: {e}")
                        print(f"Response content: {response.text[:500]}")
                        raise Exception(f"API returned invalid JSON: {e}")

                    # Check response structure and provide detailed error info
                    if "choices" not in result:
                        print(f"API response missing 'choices' field")
                        print(f"Full response: {json.dumps(result, ensure_ascii=False, indent=2)}")
                        raise Exception(f"API response format error: missing 'choices' field. Response content: {result}")

                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})
                    # Add max_completion_tokens info to usage
                    if isinstance(usage, dict):
                        usage["max_completion_tokens"] = usage.get("completion_tokens", 0)
                        # Surface whether the reply ended naturally ('stop')
                        # or was cut off by the token budget ('length').
                        usage["finish_reason"] = result["choices"][0].get("finish_reason")
                        usage["total_wait_time"] = total_wait_time
                    return content, usage
                elif self._is_retryable_status(response.status_code):
                    wait_time = self._get_retry_wait_time(attempt, base_wait_time, max_wait_time)
                    error_brief = self._safe_error_text(response)
                    last_error = f"API request failed: {response.status_code} - {error_brief}"
                    reason = "429 rate limit error" if response.status_code == 429 else f"HTTP {response.status_code} error"
                    if attempt < max_retries - 1:
                        print(f"Encountered {reason}, waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        total_wait_time += wait_time
                        continue
                    raise Exception(f"API request failed: reached max retries {max_retries}, last error: {last_error}")
                else:
                    # vLLM/local servers reject max_tokens larger than the
                    # remaining context window with a 400; shrink it to fit
                    # and retry instead of failing the whole call.
                    if self._cap_max_tokens_for_context(data, response):
                        continue
                    raise Exception(f"API request failed: {response.status_code} - {self._safe_error_text(response)}")

            except requests.exceptions.RequestException as e:
                # Record specific network error type
                error_type = type(e).__name__
                last_error = f"Network request failed ({error_type}): {e}"

                # For network errors, also use longer wait time
                wait_time = self._get_retry_wait_time(attempt, base_wait_time, max_wait_time)

                if attempt < max_retries - 1:
                    print(f"Network request failed ({error_type}), waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    total_wait_time += wait_time
                    continue
                else:
                    raise Exception(f"Network request failed: reached max retries {max_retries} - {e}")

        # If all retries fail, raise the last error
        raise Exception(last_error)

    def _call_api_with_retry_text(self, prompt: str, config: Dict[str, Any] = None) -> tuple:
        """Call text API with retry, return (result, usage)"""
        if config is None:
            config = self.config

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.get('api_key')}"
        }

        data = {
            "model": config.get("model"),
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": config.get("max_tokens", 2000),
            "temperature": config.get("temperature", 0.1),
            "top_p": config.get("top_p", 0.95),
            "n": config.get("n", 1),
        }

        # Only send stop when explicitly provided as non-empty in config; some models (e.g. gpt-5-mini) do not support this parameter
        stop_value = config.get("stop")
        if stop_value:
            data["stop"] = stop_value

        # Only add thinking parameter when it exists in the config file
        if "thinking" in config:
            data["thinking"] = config["thinking"]

        # Optional sampling knobs (e.g. repetition_penalty for local vLLM
        # servers where small models degenerate into repetition loops);
        # forwarded only when explicitly present in the config.
        for opt_key in ("repetition_penalty", "frequency_penalty",
                        "presence_penalty", "top_k", "min_p"):
            if config.get(opt_key) is not None:
                data[opt_key] = config[opt_key]

        # Timeout: separate connect and read; faster retry on connect failure
        timeout = self._get_request_timeout(config)

        if self._should_stream(config):
            return self._stream_chat_completion(data, headers, timeout, config)

        # Increase retry count and exponential backoff
        max_retries = config.get("max_retries", 8)
        base_wait_time = config.get("base_wait_time", 20)
        max_wait_time = config.get("max_wait_time", 120)
        last_error = None
        total_api_latency = 0  # Track pure API request latency (excluding wait time)
        total_wait_time = 0    # Track retry wait time

        for attempt in range(max_retries):
            # Enforce request interval
            time.sleep(2)
            total_wait_time += 2

            # Send request
            try:
                req_start_time = time.time()
                response = self.session.post(
                    f"{config.get('base_url', '').rstrip('/')}/chat/completions",
                    headers=headers,
                    json=data,
                    timeout=timeout
                )
                req_end_time = time.time()
                total_api_latency += (req_end_time - req_start_time)

                if response.status_code == 200:
                    # Check if response content is empty
                    if not response.text.strip():
                        raise Exception("API returned empty response")

                    # Try to parse JSON
                    try:
                        result = response.json()
                    except json.JSONDecodeError as e:
                        print(f"JSON parse failed: {e}")
                        print(f"Response content: {response.text[:500]}")
                        raise Exception(f"API returned invalid JSON: {e}")

                    # Check response structure and provide detailed error info
                    if "choices" not in result:
                        print(f"API response missing 'choices' field")
                        print(f"Full response: {json.dumps(result, ensure_ascii=False, indent=2)}")
                        raise Exception(f"API response format error: missing 'choices' field. Response content: {result}")

                    content = result["choices"][0]["message"]["content"]
                    usage = result.get("usage", {})
                    # Add max_completion_tokens info to usage
                    if isinstance(usage, dict):
                        usage["max_completion_tokens"] = usage.get("completion_tokens", 0)
                        # Surface whether the reply ended naturally ('stop')
                        # or was cut off by the token budget ('length').
                        usage["finish_reason"] = result["choices"][0].get("finish_reason")
                        # Add latency info
                        usage["api_latency"] = total_api_latency
                        usage["total_wait_time"] = total_wait_time
                    return content, usage
                elif self._is_retryable_status(response.status_code):
                    wait_time = self._get_retry_wait_time(attempt, base_wait_time, max_wait_time)
                    error_brief = self._safe_error_text(response)
                    last_error = f"API request failed: {response.status_code} - {error_brief}"
                    reason = "429 rate limit error" if response.status_code == 429 else f"HTTP {response.status_code} error"
                    if attempt < max_retries - 1:
                        print(f"Encountered {reason}, waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                        time.sleep(wait_time)
                        total_wait_time += wait_time
                        continue
                    raise Exception(f"API request failed: reached max retries {max_retries}, last error: {last_error}")
                else:
                    # vLLM/local servers reject max_tokens larger than the
                    # remaining context window with a 400; shrink it to fit
                    # and retry instead of failing the whole call.
                    if self._cap_max_tokens_for_context(data, response):
                        continue
                    raise Exception(f"API request failed: {response.status_code} - {self._safe_error_text(response)}")

            except requests.exceptions.RequestException as e:
                # If it's a timeout exception, also record the elapsed time (if it was a read timeout)
                # Simple handling here, only count complete request-response cycles; interrupted ones are not counted (or cannot be accurately calculated)
                # But for rigor, we assume the request has ended when an exception is thrown
                req_end_time = time.time()
                # Only accumulate if this request time hasn't been recorded yet
                # (when requests throws an exception, the total_api_latency += ... above hasn't executed yet)
                # But note req_start_time must already be defined
                try:
                    total_api_latency += (req_end_time - req_start_time)
                except UnboundLocalError:
                    pass

                error_type = type(e).__name__
                last_error = f"Network request failed ({error_type}): {e}"
                wait_time = self._get_retry_wait_time(attempt, base_wait_time, max_wait_time)

                if attempt < max_retries - 1:
                    print(f"Network request failed ({error_type}), waiting {wait_time} seconds before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    total_wait_time += wait_time
                    continue
                else:
                    raise Exception(f"Network request failed: reached max retries {max_retries} - {e}")

        # If all retries fail, raise the last error
        raise Exception(last_error)

    def test_connection(self) -> bool:
        """Test API connection"""
        try:
            # Test text generation
            test_prompt = "Please answer: what is 1+1?"
            response = self.generate_text(test_prompt)
            print(f"API connection test successful, response: {response[:50]}...")
            return True
        except Exception as e:
            print(f"API connection test failed: {e}")
            return False


def test_vision_api_client():
    """Test the vision API client"""
    print("=== Testing Vision API Client ===")
    
    # Create client
    client = VisionAPIClient()
    
    # Test connection
    if client.test_connection():
        print("API client initialization successful")
        
        # If there is a test image, can test image analysis
        test_image_path = "data/Algorithm/images/weekly-contest-381-minimum-number-of-pushes-to-type-word-i.png"
        
        if Path(test_image_path).exists():
            print(f"\nTesting image analysis: {test_image_path}")
            
            try:
                with open(test_image_path, 'rb') as f:
                    image_data = f.read()
                
                prompt = "Please briefly describe the content of this flowchart"
                result = client.analyze_image(image_data, prompt)
                print(f"Analysis result: {result[:200]}...")
                
            except Exception as e:
                print(f"Image analysis test failed: {e}")
        else:
            print(f"Test image does not exist: {test_image_path}")
    else:
        print("API client initialization failed")


if __name__ == "__main__":
    test_vision_api_client()
