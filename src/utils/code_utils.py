"""
Code utility functions
Contains code extraction and cleaning utility functions, consistent with src
"""

import re


def extract_code(content: str) -> str:
    """
    Extract code from response content (consistent with src)

    Args:
        content: Response content, may contain markdown code blocks

    Returns:
        Extracted code
    """
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


def construct_prompt(starter_code: str) -> str:
    """
    Build code generation prompt (consistent with src)

    Args:
        starter_code: Starter code

    Returns:
        Prompt string
    """
    prompt = """Generate code according to flowchart.
Note: If you're using a specific python package, you'll need to import it yourself. You don't need to use functions like input() to get input, just complete the python function.
Starter Code:
```python
%%%starter_code%%%
```
Present the code between ```python and ```."""
    return prompt.replace("%%%starter_code%%%", starter_code)


def construct_prompt_mask(starter_code: str) -> str:
    """
    Build masked code generation prompt (consistent with src)

    Args:
        starter_code: Starter code

    Returns:
        Prompt string
    """
    prompt = """Generate code according to flowchart.
Note: If you're using a specific python package, you'll need to import it yourself. You don't need to use functions like input() to get input, just complete the python function.
Note: A small part of the flowchart may be masked, you need to understand the flowchart and then generate the complete code.
Starter Code:
```python
%%%starter_code%%%
```
Present the code between ```python and ```."""
    return prompt.replace("%%%starter_code%%%", starter_code)


if __name__ == "__main__":
    # Test extract_code function
    test_content = """Here is the code:

```python
def add(a, b):
    return a + b
assert add(1, 2) == 3
```

This should work."""

    print("=== Test extract_code ===")
    print("Original content:")
    print(test_content)
    print("\nExtracted code:")
    print(extract_code(test_content))

    # Test construct_prompt
    print("\n=== Test construct_prompt ===")
    starter = "def solution():\n    pass"
    prompt = construct_prompt(starter)
    print("Generated prompt:")
    print(prompt)