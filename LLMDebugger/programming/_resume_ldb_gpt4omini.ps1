$ErrorActionPreference = "Stop"
$env:LDB_CONFIG_PATH = "F:\科研\flowchart\flowchart2code\src\configs\gpt_api_key_config.json"
$env:LDB_SKIP_INLINE_EVAL = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
Set-Location "F:\科研\flowchart\flowchart2code"
& "D:\anaconda3\python.exe" -u "LLMDebugger\programming\run_ldb_f2c_datasets.py" --config "src\configs\gpt_api_key_config.json"
exit $LASTEXITCODE
