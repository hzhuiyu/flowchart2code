
import re
import argparse
import json

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

def calculate_token_statistics(results):
    """
    Calculate overall token statistics
    """
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    valid_count = 0

    for result in results:
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
def Score(results):
    """
    score function
    easy, medium, hard
    """
    keys = results[0].keys()
    res_len = len(results)
    easy, medium, hard = [], [], []
    if "meta" in keys:
        for line in results:
            if line["meta"]["difficulty"].lower() == "easy":
                easy.append(line)
            elif line["meta"]["difficulty"].lower() == "medium":
                medium.append(line)
            elif line["meta"]["difficulty"].lower() == "hard":
                hard.append(line)
    elif "difficulty" in keys:
        for line in results:
            if line["difficulty"].lower() == "easy":
                easy.append(line)
            elif line["difficulty"].lower() == "medium":
                medium.append(line)
            elif line["difficulty"].lower() == "hard":
                hard.append(line)
    else:
        return ""

    easy_success = 0
    for line in easy:
        if line["passed"]:
            easy_success += 1
    easy_score = easy_success / len(easy)

    medium_success = 0
    for line in medium:
        if line["passed"]:
            medium_success += 1
    medium_score = medium_success / len(medium)

    hard_success = 0
    for line in hard:
        if line["passed"]:
            hard_success += 1
    hard_score = hard_success / len(hard)

    # Calculate token statistics within difficulty levels
    def calculate_token_stats(items):
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        valid_count = 0

        for item in items:
            usage = item.get("usage", {})
            if isinstance(usage, dict) and usage:
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens_usage = usage.get("total_tokens", 0)

                total_prompt_tokens += prompt_tokens
                total_completion_tokens += completion_tokens
                total_tokens += total_tokens_usage
                valid_count += 1

        if valid_count > 0:
            return {
                "count": valid_count,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "avg_prompt_tokens": total_prompt_tokens / valid_count,
                "avg_completion_tokens": total_completion_tokens / valid_count,
                "avg_total_tokens": total_tokens / valid_count
            }
        return None

    easy_token_stats = calculate_token_stats(easy)
    medium_token_stats = calculate_token_stats(medium)
    hard_token_stats = calculate_token_stats(hard)

    # Return detailed calculation process in a string
    result_str = f"\n\nEasy: {easy_success}/{len(easy)} = {easy_score}\nMedium: {medium_success}/{len(medium)} = {medium_score}\nHard: {hard_success}/{len(hard)} = {hard_score}"

    # Add token statistics info
    if easy_token_stats:
        result_str += f"\n\nEasy Token Stats: Avg {easy_token_stats['avg_total_tokens']:.2f} tokens/question"
    if medium_token_stats:
        result_str += f"\nMedium Token Stats: Avg {medium_token_stats['avg_total_tokens']:.2f} tokens/question"
    if hard_token_stats:
        result_str += f"\nHard Token Stats: Avg {hard_token_stats['avg_total_tokens']:.2f} tokens/question"

    return result_str
    

# main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Path to the results file")
    parser.add_argument("--token_only", action="store_true", help="Only show token statistics")
    args = parser.parse_args()

    results = read_jsonl_file(args.file)

    if args.token_only:
        # Only show token statistics
        token_stats = calculate_token_statistics(results)
        if token_stats:
            print("=" * 60)
            print("Token Statistics")
            print("=" * 60)
            print(f"Total tasks: {token_stats['total_tasks']}")
            print(f"Total prompt tokens: {token_stats['total_prompt_tokens']}")
            print(f"Total completion tokens: {token_stats['total_completion_tokens']}")
            print(f"Total tokens: {token_stats['total_tokens']}")
            print(f"Average prompt tokens per task: {token_stats['avg_prompt_tokens']:.2f}")
            print(f"Average completion tokens per task: {token_stats['avg_completion_tokens']:.2f}")
            print(f"Average total tokens per task: {token_stats['avg_total_tokens']:.2f}")
        else:
            print("No valid token data found.")
    else:
        # Show scores and token statistics
        print(Score(results))
        token_stats = calculate_token_statistics(results)
        if token_stats:
            print("\n" + "=" * 60)
            print("Overall Token Statistics")
            print("=" * 60)
            print(f"Total tasks: {token_stats['total_tasks']}")
            print(f"Average total tokens per task: {token_stats['avg_total_tokens']:.2f}")

    