# Stage-Attribution Controlled Set and Error-Classification Evaluation

Model run: `Qwen3-VL-8B-Instruct-Qwen3-VL-8B-Instruct` (initial generation,
no reflection). This directory supports the paper's stage-attribution study
(Table 4): a balanced controlled set with unambiguous plan/code error ground
truth, plus the classification results of the framework's attribution
pipeline over it.

## 1. Controlled Set Construction

### 1.1 Initially failed samples

Extracted from `output/<dataset>/Qwen3-VL-8B-Instruct-Qwen3-VL-8B-Instruct/
samples.jsonl_results.jsonl` where `passed == false`, keeping code, ILR, test
cases and the cached OCR data.

- `failed_samples.json` — **111** initially failed samples
  (Algorithm 51 / HumanEval-V 28 / MATH 32).

### 1.2 Controlled set (first 50, in merged order = first 50 Algorithm failures)

For every sample two artifacts were produced (`controlled_set.json`):

1. **Repaired ILR** (`fixed_ilr`): fixed until it passes the sandboxed ILR
   interpreter on **all** test cases (100/100 per sample). Repairs cover
   broken control-flow links, missing end nodes, invalid JSON, infinite
   loops, swapped argument unpacking, unsupported builtins, and node-budget
   rewrites. Test-case hardcoding is not allowed.
2. **Modified code** (`modified_code`): a direct `class Solution`
   translation of the repaired ILR with exactly **one** injected
   implementation-level bug (off-by-one bound, flipped comparison, wrong
   initialization, wrong variable updated, ...), so the program fails while
   the plan stays correct. Each bug is documented in `bug_injected`; ILR
   repairs in `ilr_fix_log`.

Verification (`controlled_set_verification.json`): all 50 repaired ILRs pass
100/100; all modified programs fail. Of the original ILRs, 49 fail the
interpreter and 1 (`check-if-array-is-good`) already passes (its original
failure was genuinely a code error).

### 1.3 Classification input (`classification_input.json`, 100 entries)

Two variants per sample:

| variant | ILR | code | ground_truth |
|---|---|---|---|
| `code_error` | fixed_ilr (interpreter-verified) | modified_code (injected bug) | code (50) |
| `ilr_error` | original_ilr | original_code | ilr (49) / code (1, check-if-array-is-good) |

## 2. Attribution Pipeline (final)

`classification_results.jsonl` holds the framework's attribution over all 100
entries (paper Table 4). The decision procedure:

1. **Deterministic primary gate**: run the ILR through the sandboxed
   interpreter on all test cases. If it passes every case, go straight to the
   code branch — the OCR label comparison (containment metric) and the LLM
   are never allowed to override a verified-correct ILR.
2. **Localization branch** (ILR fails): the OCR comparison and the LLM
   localize the origin (flowchart text extraction vs logic extraction); the
   prompt premise matches the deterministic evidence. If the LLM answers
   NO_ISSUE, fall through to the code branch (escape hatch).
3. **Code branch**: the code test failures are summarized (public-test-first
   context; hidden-test details withheld while all public cases pass) and the
   LLM analyzes the code => "code".

**Coverage-gap safeguard**: when the ILR passed the interpreter, the Critic
may still reattribute a failure to the plan only by emitting
`ILR_LOGIC_ISSUE: node <id> | input: <literal> | ilr_output: <literal> | expected: <literal>`
which must survive three deterministic arbiters (input not covered by the
passed test suite; the LLM's hand-trace matches actual ILR execution; actual
output differs from the claimed expected output).

## 3. Results (100/100 entries)

| group | n | correct | accuracy |
|---|---|---|---|
| code_error variant | 50 | 50 | **100.0%** |
| ilr_error variant | 50 | 49 | 98.0% |
| overall | 100 | 99 | **99.0%** |

Decision paths: 50 `ilr_test_passed` (straight to code, all correct), 49
`ilr_test_failed+llm_confirmed` (all correct), 1
`ilr_test_failed+llm_overridden` (LLM claimed an interpreter limitation for
an ILR that calls a non-existent `list.bisect_left` method => misattributed
to code). The single misclassification is the retained cost of the NO_ISSUE
escape hatch; disabling the override would yield 100/100.

## 4. File Inventory

- `failed_samples.json` — 111 initially failed samples (original ILR + code + tests + OCR cache)
- `baseline_ilr_report.json` — interpreter baseline of the 111 original ILRs (pre-repair)
- `controlled_set.json` — 50 samples with fixed_ilr / modified_code / ilr_fix_log / bug_injected
- `controlled_set_verification.json` — verification records for the 50 samples
- `classification_input.json` — 100 classification entries with ground truth
- `classification_results.jsonl` — attribution results (per-entry prediction, ILR test result, LLM analysis, decision path, usage)
- `batches/` — per-batch working files used while repairing the ILRs and injecting the code bugs (batch_0..4 + per-sample fix artifacts + verification reports)

## 5. Notes

- The ILR interpreter puts the test-case list objects directly into `ctx`;
  some ILRs mutate their input arguments in place (e.g. `grid[i][j] = ...`).
  All tooling therefore deep-copies test cases before execution; keep that
  practice in any new script or dataset files get silently corrupted.
- `check-if-array-is-good::ilr_error` is a designed edge: the original ILR is
  correct (truth=code) and the framework's answer is included in the overall
  accuracy.
