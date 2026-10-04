from .py_generate import PyGenerator
from .model import CodeLlama, ModelBase, GPT4, GPT35, StarCoder, OpenAICompatChat, load_ldb_config

def model_factory(model_name: str, port: str = "", key: str = "") -> ModelBase:
    # LDB config file (e.g. flowchart2code's gpt_api_key_config.json) takes
    # precedence: it supplies api_key/base_url/model/max_tokens in one place.
    if load_ldb_config():
        return OpenAICompatChat(model_name, key)
    if "gpt-4" in model_name or "gpt-5" in model_name or model_name.startswith("o3") or model_name.startswith("o4"):
        return GPT4(model_name, key)
    elif "gpt-3.5" in model_name:
        return GPT35(model_name, key)
    elif model_name == "starcoder":
        return StarCoder(port)
    elif model_name == "codellama":
        return CodeLlama(port)
    else:
        # Any OpenAI-compatible chat endpoint (vLLM server, relay, etc.);
        # configure via LDB_OPENAI_BASE_URL / LDB_OPENAI_API_KEY.
        return OpenAICompatChat(model_name, key)
