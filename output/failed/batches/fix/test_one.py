"""Direct per-sample tester: reads the extracted sample json plus candidate
ILR/code files, so iteration does not require updating the batch file."""
import json
import sys
from pathlib import Path

ROOT = Path(r"F:\科研\flowchart\flowchart2code")
sys.path.insert(0, str(ROOT / "src"))

from tools.ilr_interpreter import ILRTester  # noqa: E402
from tools.code_tester import CodeTester  # noqa: E402


def summarize(result, max_detail=3):
    total = result.get("total", 0)
    passed = result.get("passed", 0)
    line = f"{passed}/{total}"
    if result.get("error"):
        err = str(result["error"]).replace("\n", " ")[:300]
        line += f" ERROR: {err}"
    else:
        failed = [r for r in result.get("results", []) if not r.get("passed")]
        for t in failed[:max_detail]:
            line += f" | case{t.get('test_num')}: expected={t.get('expected_output')!r} actual={t.get('actual_output')!r}"
            if t.get("error"):
                line += f" ({str(t['error'])[:200]})"
    return line


def main():
    sample_path = sys.argv[1]
    ilr_path = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None
    code_path = sys.argv[3] if len(sys.argv) > 3 else None
    s = json.load(open(sample_path, encoding="utf-8"))

    if ilr_path:
        ilr = Path(ilr_path).read_text(encoding="utf-8")
        r = ILRTester().test_ilr(ilr_json=ilr, test_cases=s["test_cases"],
                                 entry_point=s.get("entry_point") or None)
        print("ILR :", summarize(r))
        if ilr_path.endswith("orig"):
            pass

    if code_path:
        code = Path(code_path).read_text(encoding="utf-8")
        info = {
            "task_id": s["task_id"], "test_cases": s["test_cases"],
            "entry_point": s.get("entry_point") or "", "starter_code": s.get("starter_code") or "",
            "test": s.get("test", ""), "prompt": s.get("prompt", ""), "dataset": s.get("dataset", ""),
        }
        r = CodeTester().test_generated_code(code, info)
        print("CODE:", summarize(r))


if __name__ == "__main__":
    main()
