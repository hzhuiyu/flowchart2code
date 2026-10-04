from typing import List, Union, Optional, Literal
import dataclasses
import json
import os
try:
    from vllm import LLM, SamplingParams  # noqa: F401  (only needed for local vLLM servers; not available on Windows)
except Exception:  # ImportError or broken transitive imports
    LLM = None
    SamplingParams = None
from tenacity import (
    retry,
    stop_after_attempt,  # type: ignore
    wait_random_exponential,  # type: ignore
)
from openai import OpenAI

try:
    from transformers import GPT2Tokenizer, AutoTokenizer
except Exception:
    # torch DLL/import failures surface as OSError on some Windows setups;
    # token counting then falls back to whitespace splitting.
    GPT2Tokenizer = None
    AutoTokenizer = None

from flowchart_image import has_image, message_text

MessageRole = Literal["system", "user", "assistant"]

# Context budget for chat models; raise via env for long LCB problem statements.
# Image runs need a much larger window (flowchart + prompt); default 3097 would
# drop the multimodal user turn.
_DEFAULT_MAX_CONTEXT = "100000" if os.getenv("LDB_USE_IMAGE", "").strip().lower() in (
    "1", "true", "yes", "on"
) else "3097"
MAX_CONTEXT_TOKENS = int(os.getenv("LDB_MAX_CONTEXT_TOKENS", _DEFAULT_MAX_CONTEXT))


def _looks_like_repetition_loop(text: str) -> bool:
    """Cheap detector for degenerate decode loops (repeated suffix / lines)."""
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


def load_ldb_config() -> dict:
    """flowchart2code-style API config (api_key/base_url/model/max_tokens).

    Enabled by pointing LDB_CONFIG_PATH at e.g.
    flowchart2code/src/configs/gpt_api_key_config.json.
    """
    path = os.getenv("LDB_CONFIG_PATH", "")
    if not path or not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return cfg if isinstance(cfg, dict) else {}
    except Exception as e:
        print("LDB config load failed:", e)
        return {}

@dataclasses.dataclass()
class Message():
    role: MessageRole
    # str for text-only turns; OpenAI multimodal list for flowchart image turns.
    content: Union[str, list]


def message_to_str(message: Message) -> str:
    return f"{message.role}: {message_text(message.content)}"


def messages_to_str(messages: List[Message]) -> str:
    return "\n".join([message_to_str(message) for message in messages])


@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def gpt_completion(
    model: str,
    prompt: str,
    max_tokens: int = 1024,
    stop_strs: Optional[List[str]] = None,
    temperature: float = 0.0,
    num_comps=1,
) -> Union[List[str], str]:
    response = client.chat.completions.create(
        model=model,
        messages=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
        n=num_comps,
    )
    if num_comps == 1:
        return response.choices[0].text  # type: ignore

    return [choice.text for choice in response.choices]  # type: ignore


def _tok_len(tokenizer, text: str) -> int:
    if tokenizer is None:
        return len(text.split())
    return len(tokenizer.tokenize(text))


def change_messages(tokenizer, messages, max_len):
    if isinstance(messages, str):
        message_lines = messages.split("\n")
        acc_msg_len = 0
        new_messages = ""
        for l in reversed(message_lines):
            acc_msg_len += _tok_len(tokenizer, l)
            if acc_msg_len < max_len:
                new_messages = l + "\n" + new_messages
            else:
                break
        new_messages = new_messages.strip()
        return new_messages
    else:
        def _msg_len(msg) -> int:
            return _tok_len(tokenizer, message_text(msg.content))

        # Always keep the leading system prompt and any multimodal (image)
        # user turn. Trimming those would silently drop the flowchart.
        keep_prefix = []
        rest = list(messages)
        if rest:
            keep_prefix.append(rest.pop(0))
        while rest and has_image(rest[0].content):
            keep_prefix.append(rest.pop(0))
        total_msg_len = sum(_msg_len(msg) for msg in keep_prefix)
        rest_messages = []
        for msg in reversed(rest):
            msg_len = _msg_len(msg)
            if msg_len + total_msg_len < max_len:
                rest_messages = [msg] + rest_messages
                total_msg_len += msg_len
            else:
                break
        messages = keep_prefix + rest_messages
    return messages

class ModelBase():
    def __init__(self, name: str):
        self.name = name
        self.is_chat = False

    def __repr__(self) -> str:
        return f'{self.name}'

    def generate_chat(self, messages: List[Message], max_tokens: int = 1024, temperature: float = 0.2, num_comps: int = 1) -> Union[List[str], str]:
        raise NotImplementedError

    def generate(self, prompt: str, max_tokens: int = 1024, stop_strs: Optional[List[str]] = None, temperature: float = 0.0, num_comps=1) -> Union[List[str], str]:
        raise NotImplementedError


class GPTChat(ModelBase):
    def __init__(self, model_name: str, key: str = "", base_url: str = ""):
        self.name = model_name
        self.is_chat = True
        # Subclasses (OpenAICompatChat) may have resolved this from the config
        # file before calling super().
        if not hasattr(self, "default_max_tokens"):
            self.default_max_tokens = int(os.getenv("LDB_MAX_TOKENS", "1024"))
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2") if GPT2Tokenizer is not None else None
        kwargs = {}
        if key != "":
            kwargs["api_key"] = key
        if base_url != "":
            kwargs["base_url"] = base_url
        # Relays are slow; never hang forever on a single request.
        # Vision turns are larger; image runs should raise LDB_API_TIMEOUT_SEC.
        kwargs["timeout"] = float(os.getenv("LDB_API_TIMEOUT_SEC", "300"))
        self.client = OpenAI(**kwargs)
    
    def gpt_chat(
        self,
        messages,
        stop: List[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        num_comps=1,
    ) -> Union[List[str], str]:
        import time as _time
        create_kwargs = dict(
            model=self.name,
            temperature=temperature,
            top_p=getattr(self, "top_p", 1),
            n=num_comps,
        )
        freq_pen = getattr(self, "frequency_penalty", None)
        if freq_pen is not None:
            create_kwargs["frequency_penalty"] = freq_pen
        pres_pen = getattr(self, "presence_penalty", None)
        if pres_pen is not None:
            create_kwargs["presence_penalty"] = pres_pen
        extra_body = {}
        rep_pen = getattr(self, "repetition_penalty", None)
        if rep_pen is not None:
            extra_body["repetition_penalty"] = rep_pen
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        if max_tokens is not None:
            create_kwargs["max_tokens"] = max_tokens
        max_trials = 6
        trial = 0
        # Track whether this endpoint supports the `stop` parameter.
        # Once we detect it's unsupported, we stop sending it entirely.
        _stop_supported = True
        # First-token wait can be long on VL / busy relays; mid-stream idle
        # is the "connection accepted but stalled" detector.
        _stream_idle_timeout = float(os.getenv(
            "LDB_STREAM_IDLE_TIMEOUT",
            str(getattr(self, "stream_idle_timeout", 120)),
        ))
        _stream_first_timeout = float(os.getenv(
            "LDB_STREAM_FIRST_TIMEOUT",
            str(getattr(self, "stream_first_timeout", 300)),
        ))
        while True:
            try:
                new_messages = change_messages(self.tokenizer, messages, MAX_CONTEXT_TOKENS)
                messages = new_messages
                request_kwargs = dict(
                    messages=[dataclasses.asdict(message) for message in messages],
                    **create_kwargs,
                )
                if stop and _stop_supported:
                    request_kwargs["stop"] = stop
                # Use streaming so we can detect stalls (connection accepted
                # but no data arriving) instead of waiting for the full
                # timeout on a single non-streaming response.
                request_kwargs["stream"] = True
                stream = self.client.chat.completions.create(**request_kwargs)
                import threading as _threading
                collected_content = []
                chunk_error = [None]
                chunk_done = _threading.Event()
                last_activity = [_time.monotonic()]
                got_chunk = [False]

                max_output_chars = int(getattr(self, "max_output_chars", 0) or 0)
                loop_abort = [False]

                def _consume_stream():
                    try:
                        chars_since_check = 0
                        for chunk in stream:
                            last_activity[0] = _time.monotonic()
                            got_chunk[0] = True
                            if chunk.choices and chunk.choices[0].delta.content:
                                piece = chunk.choices[0].delta.content
                                collected_content.append(piece)
                                chars_since_check += len(piece)
                                nchars = sum(len(x) for x in collected_content)
                                if max_output_chars and nchars >= max_output_chars:
                                    print(f"[stream] hit max_output_chars={max_output_chars}, stopping")
                                    loop_abort[0] = True
                                    break
                                if chars_since_check >= 400:
                                    chars_since_check = 0
                                    if _looks_like_repetition_loop("".join(collected_content)):
                                        print(f"[stream] repetition loop at {nchars} chars, stopping")
                                        loop_abort[0] = True
                                        break
                    except Exception as exc:
                        chunk_error[0] = exc
                    finally:
                        chunk_done.set()

                reader = _threading.Thread(target=_consume_stream, daemon=True)
                reader.start()
                last_report = _time.monotonic()
                while not chunk_done.wait(timeout=1.0):
                    idle = _time.monotonic() - last_activity[0]
                    now = _time.monotonic()
                    limit = _stream_idle_timeout if got_chunk[0] else _stream_first_timeout
                    if now - last_report >= 30:
                        nchars = sum(len(x) for x in collected_content)
                        phase = "streaming" if got_chunk[0] else "waiting first token"
                        print(f"[stream] {phase}: {nchars} chars, idle {idle:.0f}s / {limit:.0f}s")
                        last_report = now
                    if idle >= limit:
                        nchars = sum(len(x) for x in collected_content)
                        phase = "mid-stream" if got_chunk[0] else "first-token"
                        print(f"Stream idle for {idle:.0f}s ({phase}, "
                              f"chunks={len(collected_content)}, chars={nchars}), treating as timeout")
                        chunk_error[0] = TimeoutError(f"No data received for {idle:.0f}s")
                        try:
                            stream.close()
                        except Exception:
                            pass
                        break
                if chunk_error[0] is not None:
                    raise chunk_error[0]
                full_content = "".join(collected_content)
                break
            except Exception as e:
                print("GPT Error:", str(e))
                if "context_length_exceeded" in str(e) or "maximum context length" in str(e) or "context window" in str(e).lower():
                    # Shrink history and retry immediately (no trial consumed).
                    messages = change_messages(self.tokenizer, messages, max(1024, MAX_CONTEXT_TOKENS // 2))
                    print("AFTER CHANGE MESSAGE LEN:", len(messages))
                    continue
                if "stop" in str(e).lower() and ("unsupported" in str(e).lower() or "not supported" in str(e).lower()):
                    # Some relay endpoints (e.g. Gemini proxies) don't support
                    # the `stop` parameter.  Drop it and retry immediately.
                    print("stop parameter not supported by this endpoint; retrying without stop")
                    _stop_supported = False
                    stop = None
                    continue
                err = str(e).lower()
                dropped = False
                for bad_key in ("repetition_penalty", "frequency_penalty", "presence_penalty", "extra_body"):
                    if bad_key in err and ("unexpected" in err or "unknown" in err or "not supported" in err or "unsupported" in err):
                        create_kwargs.pop(bad_key, None)
                        if bad_key == "extra_body" or bad_key == "repetition_penalty":
                            create_kwargs.pop("extra_body", None)
                        print(f"{bad_key} not supported; retrying without it")
                        dropped = True
                if dropped:
                    continue
                trial += 1
                if trial >= max_trials:
                    # After all retries exhausted, return an empty string so
                    # the caller can gracefully skip this sample instead of
                    # crashing the entire LDB run.
                    print(f"All {max_trials} retries exhausted; returning empty response")
                    return "" if num_comps == 1 else [""]
                # Transient relay failures (timeout / 429 / 5xx / connection):
                # back off and retry instead of aborting the whole run.
                wait_s = min(120, 10 * (2 ** (trial - 1)))
                print(f"Retry {trial}/{max_trials} in {wait_s}s ...")
                _time.sleep(wait_s)
        if num_comps == 1:
            return full_content
        return [full_content]

    def generate_chat(self, messages: List[Message], stop: List[str] = None, max_tokens: int = None, temperature: float = 0.0, num_comps: int = 1) -> Union[List[str], str]:
        if max_tokens is None:
            max_tokens = self.default_max_tokens
        cfg_temp = getattr(self, "default_temperature", None)
        if temperature == 0.0 and cfg_temp not in (None, 0, 0.0):
            temperature = float(cfg_temp)
        res = self.gpt_chat(messages, stop, max_tokens, temperature, num_comps)
        return res


class GPT4(GPTChat):
    def __init__(self, model, key):
        super().__init__(model, key)


class GPT35(GPTChat):
    def __init__(self, model, key):
        super().__init__(model, key)


class OpenAICompatChat(GPTChat):
    """Any OpenAI-compatible chat endpoint.

    Endpoint/credentials come from the LDB config file (LDB_CONFIG_PATH, e.g.
    flowchart2code's gpt_api_key_config.json) or env
    (LDB_OPENAI_BASE_URL / LDB_OPENAI_API_KEY) so arbitrary served models
    (vLLM, relays, ...) can drive LDB without touching the model registry.
    """

    def __init__(self, model_name: str, key: str = "", base_url: str = ""):
        cfg = load_ldb_config()
        self.default_max_tokens = int(os.getenv("LDB_MAX_TOKENS", "1024"))
        if cfg:
            key = key or cfg.get("api_key", "")
            base_url = base_url or cfg.get("base_url", "")
            if model_name in ("", "config"):
                model_name = cfg.get("model", model_name)
            try:
                self.default_max_tokens = int(cfg.get("max_tokens", self.default_max_tokens))
            except (TypeError, ValueError):
                pass
            if cfg.get("timeout_sec") and not os.getenv("LDB_API_TIMEOUT_SEC"):
                os.environ["LDB_API_TIMEOUT_SEC"] = str(cfg["timeout_sec"])
            try:
                self.stream_idle_timeout = float(cfg.get("stream_idle_timeout", 120))
            except (TypeError, ValueError):
                self.stream_idle_timeout = 120
            try:
                self.stream_first_timeout = float(cfg.get(
                    "stream_first_timeout",
                    cfg.get("timeout_sec", 300),
                ))
            except (TypeError, ValueError):
                self.stream_first_timeout = 300
            try:
                self.default_temperature = float(cfg["temperature"]) if cfg.get("temperature") is not None else None
            except (TypeError, ValueError):
                self.default_temperature = None
            try:
                self.top_p = float(cfg["top_p"]) if cfg.get("top_p") is not None else 1
            except (TypeError, ValueError):
                self.top_p = 1
            for key_name in ("frequency_penalty", "presence_penalty", "repetition_penalty"):
                if cfg.get(key_name) is not None:
                    try:
                        setattr(self, key_name, float(cfg[key_name]))
                    except (TypeError, ValueError):
                        pass
            try:
                self.max_output_chars = int(cfg.get("max_output_chars", 0) or 0)
            except (TypeError, ValueError):
                self.max_output_chars = 0
        key = key or os.getenv("LDB_OPENAI_API_KEY", "")
        base_url = base_url or os.getenv("LDB_OPENAI_BASE_URL", "")
        super().__init__(model_name, key, base_url)


class VLLMModelBase(ModelBase):
    """
    Base for huggingface chat models
    """

    def __init__(self, model, port=""):
        super().__init__(model)
        port = port or "8000"
        self.model = model
        self.vllm_client = OpenAI(api_key="EMPTY", base_url=f"http://localhost:{port}/v1")
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.max_length = 7000
    
    def vllm_chat(
        self,
        prompt: str,
        stop: List[str] = [""],
        max_tokens: int = 1024,
        temperature: float = 0.0,
        num_comps=1,
    ) -> Union[List[str], str]:
        max_length = self.max_length
        while True:
            prompt = change_messages(self.tokenizer, prompt, max_length)  # StarCoder max length
            try:
                responses = self.vllm_client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    echo=False,
                    max_tokens=max_tokens,
                    temperature=0,
                    top_p=1,
                    stop=stop,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    n=num_comps,
                )
            except Exception as e:
                print("VLLM Error:", str(e))
                if "maximum context length" in str(e):
                    max_length -= 2000
                else:
                    assert False, "VLLM API error: " + str(e)
            else:
                break
        if num_comps == 1:
            return responses.choices[0].text  # type: ignore
        return [response.choices[0].text for response in responses]  # type: ignore

    def generate_completion(self, messages: str, stop: List[str] = [""], max_tokens: int = 1024, temperature: float = 0.0, num_comps: int = 1) -> Union[List[str], str]:
        ret = self.vllm_chat(messages, stop, max_tokens, temperature, num_comps)
        return ret

    def prepare_prompt(self, messages: List[Message]):
        prompt = ""
        for i, message in enumerate(messages):
            prompt += message_text(message.content) + "\n"
            if i == len(messages) - 1:
                prompt += "\n"
        return prompt

    def extract_output(self, output: str) -> str:
        return output


class StarCoder(VLLMModelBase):
    def __init__(self, port=""):
        super().__init__("bigcode/starcoder", port)


class CodeLlama(VLLMModelBase):
    def __init__(self, port=""):
        super().__init__("codellama/CodeLlama-34b-Instruct-hf", port)
