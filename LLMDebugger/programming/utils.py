import os
import gzip
import json
import re
import openai

from typing import List

openai.api_key = os.getenv("OPENAI_API_KEY")
IMPORT_HEADER = "from typing import *\nimport math\nfrom heapq import *\nimport itertools\nimport re\nimport typing\nimport heapq\n_str=str\nimport re\nfrom sortedcontainers import SortedList, SortedDict, SortedSet\nimport sortedcontainers\n"

def prepare_function_from_seed(dataset_type, prompt, seed, entry_point):
    if dataset_type in ["HumanEval", "MBPP", "HumanEval-V"]:
        if (prompt in seed) or (('def ' + entry_point + '(') in seed):
            # It has the function header, no need to add
            cur_func_impl = seed
        else:
            cur_func_impl = prompt + "\n" + seed
        # Add auxilary function
        funcs = get_function(prompt)
        seed_funcs = [func[0] for func in get_function(seed)]
        for func in funcs:
            if func[0] not in seed_funcs:
                cur_func_impl = func[1] + "\n" + cur_func_impl
        # Add comments
        if not find_comment(cur_func_impl, entry_point):
            cur_func_impl = fix_func_impl_comments(cur_func_impl, prompt, entry_point)
    elif dataset_type in ["TransCoder"]:
        # It contains a whole program
        cur_func_impl = seed
    elif dataset_type in ["LiveCodeBench", "Algorithm", "MATH"]:
        # Completions are full programs (top-level `def` or `class Solution`);
        # use them verbatim and only embed the problem text as a docstring.
        cur_func_impl = seed
        if not find_comment_lcb(cur_func_impl, entry_point):
            cur_func_impl = insert_comment_lcb(cur_func_impl, extract_lcb_problem_text(prompt), entry_point)
    # Add import header
    if IMPORT_HEADER not in cur_func_impl:
        cur_func_impl = IMPORT_HEADER + cur_func_impl
    assert isinstance(cur_func_impl, str)
    return cur_func_impl

def fix_func_impl_comments(func_impl: str, prompt: str, entry) -> str:
    # extract comments from prompt and insert them into func_impl after the function header
    if prompt.find('\"\"\"') != -1:
        comments = prompt.split('\"\"\"')[1]
    elif prompt.find('\'\'\'') != -1:
        comments = prompt.split('\'\'\'')[1]
    # Get the function header
    func_impl_lines = func_impl.split('\n')
    for i, line in enumerate(func_impl_lines):
        if line.startswith('def') and entry in line:
            break
    # Insert comments after the function header
    func_impl_lines.insert(i+1, '    \"\"\"' + comments + '\"\"\"')
    return '\n'.join(func_impl_lines)

def insert_comment(func_impl: str, comment: str, entry: str) -> str:
    func_impl_lines = func_impl.split('\n')
    for i, line in enumerate(func_impl_lines):
        if line.startswith('def ' + entry + '('):
            break
    func_impl_lines.insert(i + 1, '    \"\"\"' + comment + '\"\"\"')
    return '\n'.join(func_impl_lines)

def remove_comment(old_block: List[str]) -> str:
    new_block = []
    old_block_lines = old_block.split('\n')
    for line in old_block_lines:
        if line.lstrip().startswith('#'):
            continue
        new_block.append(line)
    if len(new_block) == 1:
        return new_block[0]
    else:
        return '\n'.join(new_block)

def extrace_comment(prompt: str) -> str:
    if prompt.find('\"\"\"') != -1:
        comments = prompt.split('\"\"\"')[-2]
    elif prompt.find('\'\'\'') != -1:
        comments = prompt.split('\'\'\'')[-2]
    else:
        # Prompts without docstrings (e.g. LiveCodeBench prose) have no comment.
        return ""
    return comments


def extract_lcb_problem_text(prompt: str) -> str:
    """Problem statement used as the docstring for LiveCodeBench items.

    Triple-quoted comment when present, otherwise the prose before the first
    ```python block.  Sanitized so it can never terminate the docstring early.
    """
    text = extrace_comment(prompt).strip()
    if not text:
        prose = prompt.split("```python")[0]
        lines = [line.strip() for line in prose.strip().splitlines() if line.strip()]
        text = "\n".join(lines[:40])
    return text.replace('"""', "'''").replace("'''", '"""').replace('"""', "'''")


def _find_def_line(lines: List[str], entry: str) -> int:
    """Index of the `def entry(` line, matching methods indented inside a class."""
    for i, line in enumerate(lines):
        stripped = line.lstrip()
        if stripped.startswith("def ") and re.match(rf"def\s+{re.escape(entry)}\s*\(", stripped):
            return i
    return -1


def find_comment_lcb(func_impl: str, entry: str) -> bool:
    """find_comment variant that also detects docstrings of class methods."""
    lines = func_impl.split("\n")
    i = _find_def_line(lines, entry)
    if i == -1:
        return False
    for line in lines[i + 1:i + 60]:
        stripped = line.strip()
        if not stripped:
            continue
        return stripped.startswith('"""') or stripped.startswith("'''") or stripped.startswith("#")
    return False


def insert_comment_lcb(func_impl: str, comment: str, entry: str) -> str:
    """Insert a docstring right after the entry def line (class-aware)."""
    lines = func_impl.split("\n")
    i = _find_def_line(lines, entry)
    if i == -1:
        return func_impl
    raw = lines[i]
    indent = raw[: len(raw) - len(raw.lstrip())] + "    "
    doc_lines = [indent + '"""']
    for comment_line in comment.split("\n"):
        doc_lines.append((indent + comment_line).rstrip() if comment_line.strip() else "")
    doc_lines.append(indent + '"""')
    return "\n".join(lines[: i + 1] + doc_lines + lines[i + 1:])

def find_comment(func_impl: str, entry: str ) -> bool:
    func_impl_lines = func_impl.split('\n')
    for i, line in enumerate(func_impl_lines):
        if line.startswith('def ' + entry + "("):
            break
    func_body = "\n".join(func_impl_lines[i:])
    if func_body.find('\"\"\"') != -1 or func_body.find('\'\'\'') != -1:
        return True
    return False

def get_function(prompt):
    lines = prompt.split('\n')
    cur_func = ""
    funcs = []
    for i, l in enumerate(lines):
        if l.startswith("def "):
            if cur_func == "":
                cur_func = l
            else:
                funcs.append([func_name, cur_func])
                cur_func = l
            func_name = l.split("def ")[1].split("(")[0]
        elif cur_func != "":
            cur_func += "\n" + l
    return funcs

def convert_comment(translation_prompt):
    cpp_prog = translation_prompt.split("[c++]")[1].split("[python]")[0]
    commented_prog = "\'\'\'\nC++ Implementation\n" + cpp_prog.strip() + "\n\'\'\'\n"
    return commented_prog

def make_printv(verbose: bool):
    def print_v(*args, **kwargs):
        if verbose:
            kwargs["flush"] = True
            print(*args, **kwargs)
        else:
            pass
    return print_v


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def read_jsonl_map(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                items[item["task_id"]] = item
    return items

def write_jsonl(path: str, data: List[dict], append: bool = False):
    with open(path, "a" if append else "w", encoding="utf-8", newline="\n") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_jsonl_gz(path: str) -> List[dict]:
    if not path.endswith(".jsonl.gz"):
        raise ValueError(f"File `{path}` is not a jsonl.gz file.")
    with gzip.open(path, "rt") as f:
        data = [json.loads(line) for line in f]
    return data


def replace_seed_test(item, items_seed, items_test):
    if item['task_id'] in items_seed:
        item['seed'] = items_seed[item['task_id']]['solution']
        if 'is_passing' in items_seed[item['task_id']]:
            item['is_passing'] = items_seed[item['task_id']]['is_passing']
        else:
            item['is_passing'] = False
    else:
        item['seed'] = ""
    if item['task_id'] in items_test:
        item['given_tests'] = items_test[item['task_id']]['given_tests']
    else:
        item['given_tests'] = []
    return item

def enumerate_resume(dataset, results_path, seedfile = None, testfile = None):
    items_seed = {}
    items_test = {}
    if seedfile is not None:
        items_seed = read_jsonl_map(seedfile)
    if testfile is not None:
        print("testfile", testfile)
        items_test = read_jsonl_map(testfile)
    
    if not os.path.exists(results_path):
        for i, item in enumerate(dataset):
            item = replace_seed_test(item, items_seed, items_test)
            yield i, item
    else:
        count = 0
        exist_items = []
        with open(results_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                exist_items.append(item['task_id'])

        for i, item in enumerate(dataset):
            # skip items that have been processed before
            if item['task_id'] in exist_items:
                continue
            item = replace_seed_test(item, items_seed, items_test)
            yield i, item


def resume_success_count(dataset) -> int:
    count = 0
    for item in dataset:
        if "is_solved" in item and item["is_solved"]:
            count += 1
    return count

def count_solved(logpath) -> float:
    solved = 0
    count = 0
    dataset = open(logpath, "r", encoding="utf-8", errors="replace")
    for l in dataset:
        item = json.loads(l)
        count += 1
        if "is_solved" in item and item["is_solved"]:
            solved += 1
    return float(solved) / count