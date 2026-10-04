"""Quick checker: run an ILR json file against a sample's test_cases (and optionally a code file)."""
import json, sys
from pathlib import Path

ROOT = Path(r"F:\科研\flowchart\flowchart2code")
sys.path.insert(0, str(ROOT / "src"))
from tools.ilr_interpreter import ILRTester  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts_failed"))
from code_test_subprocess import test_generated_code_compat  # noqa: E402


def main():
    sample_path, ilr_path = sys.argv[1], sys.argv[2]
    code_path = sys.argv[3] if len(sys.argv) > 3 else None
    s = json.load(open(sample_path, encoding='utf-8'))
    ilr = open(ilr_path, encoding='utf-8').read()
    tester = ILRTester()
    r = tester.test_ilr(ilr_json=ilr, test_cases=s["test_cases"], entry_point=s.get("entry_point") or None)
    print(f"ILR: {r.get('passed')}/{r.get('total')}")
    if r.get("error"):
        print("ERROR:", str(r["error"])[:500])
    fails = [x for x in r.get("results", []) if not x.get("passed")]
    for f in fails[:5]:
        print("  case", f.get("test_num"), "expected", f.get("expected_output"), "actual", f.get("actual_output"), str(f.get("error"))[:200])
    if code_path:
        code = open(code_path, encoding='utf-8').read()
        info = {"task_id": s["task_id"], "test_cases": s["test_cases"], "entry_point": s.get("entry_point") or "",
                "starter_code": s.get("starter_code") or "", "test": s.get("test", ""), "prompt": s.get("prompt", ""),
                "dataset": s.get("dataset", "")}
        rc = test_generated_code_compat(code, info)
        print(f"CODE: {rc.get('passed')}/{rc.get('total')}")
        if rc.get("error"):
            print("CODE ERROR:", str(rc["error"])[:300])
        cfails = [x for x in rc.get("results", []) if not x.get("passed")]
        for f in cfails[:5]:
            print("  case", f.get("test_num"), "expected", f.get("expected_output"), "actual", f.get("actual_output"), str(f.get("error"))[:200])


if __name__ == "__main__":
    main()
