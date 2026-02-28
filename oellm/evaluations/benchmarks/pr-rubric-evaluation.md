## What is the problem?

OpenJury currently only supports pairwise winrate evaluation, where a judge LLM compares two responses head-to-head and assigns scores. Pairwise evaluation has known limitations ("Tiny Aya: Bridging Scale and Multilingual Depth", Salamanca, Kreutzer, Fadaee et al., https://arxiv.org/abs/2501.10893, 2026):

- **High variance**: small style changes can flip binary preference labels, causing large swings in average win rate on small eval sets
- **No absolute quality signal**: a model might get 90% win rate just because the competitor fails entirely, inflating results without reflecting actual quality
- **No interpretability**: no insight into which quality dimensions (accuracy, fluency, etc.) drove the judge's decision
- **Position bias**: the judge may prefer whichever response appears first or second (mitigated by `swap_mode=both`, but at 2x compute)

## How do we solve it?

We add a rubric-based evaluation mode (`--eval_mode rubric`) that scores each response **independently** on 4 criteria using a 1/3/5/7 Likert scale. The evaluation prompt ([`rubric-prompt.txt`](https://github.com/ferreirafabio/OpenJury/blob/feat/rubric-evaluation/openjury/prompts/rubric-prompt.txt)) is taken from Appendix B.3 of ("Tiny Aya: Bridging Scale and Multilingual Depth", Salamanca, Kreutzer, Fadaee et al., https://arxiv.org/abs/2501.10893, 2026) and adapted with minor changes: `{language}` placeholders replaced with language-agnostic phrasing, `chatbot` -> `response` for consistency, and template variable names adapted to OpenJury conventions.

**Criteria** (each scored 1, 3, 5, or 7):
1. Instruction Following
2. Naturalness
3. Coherence
4. Accuracy

**Composite score**: mean of the 4 criteria, linearly mapped from [1, 7] to [0, 1] via `(mean - 1) / 6`.

Because rubric mode evaluates one response at a time (not pairwise), it eliminates position bias entirely. The judge prompt explicitly instructs scoring on valid anchor points only (1, 3, 5, 7); if the judge outputs an intermediate value (2, 4, 6), the parser snaps to the nearest valid anchor and logs the count.

Default mode remains `winrate` -- fully backward compatible.

**Example output** (OLMo3-7B Think SFT, alpaca-eval, Qwen3-30B-A3B judge):

```
============================================================
                 RUBRIC EVALUATION RESULTS
Dataset: alpaca-eval
Judge: Qwen/Qwen3-30B-A3B-Instruct-2507
------------------------------------------------------------
  Model A: Olmo-3-7B-Think-SFT (baseline)
  Model B: dolci-think-sft-hf-65k (ours)

  Criterion                    Model A    Model B
  -----------------------------------------------
  Instruction Following           5.99       6.26
  Naturalness                     6.70       6.83
  Coherence                       6.41       6.68
  Accuracy                        6.19       6.39
  -----------------------------------------------
  Composite (0-1)                0.887      0.923

  Evaluations: 805 | Parse failures: A=0, B=1
============================================================
```

## Changes

- `openjury/evaluate.py`: add `RUBRIC_CRITERIA`, `VALID_RUBRIC_SCORES`, `RubricScore` parser (with snap-to-valid logic), `RubricAnnotation` dataclass, `load_rubric_prompts()`, `annotate_rubric()` function
- `openjury/generate_and_evaluate.py`: add `--eval_mode` CLI argument to `CliArgs`, branch `main()` on eval mode, add `print_rubric_results()`, add `"eval_mode"` field to winrate results JSON for consistency
- `openjury/prompts/rubric-prompt.txt`: rubric user prompt with 4 criteria, adapted from Tiny Aya Appendix B.3
- `openjury/prompts/rubric-system-prompt.txt`: rubric system prompt
- `tests/test_rubric.py`: 12 tests covering JSON parsing, snapping, composites, edge cases, and end-to-end with Dummy models

## Testing

- All 12 rubric tests pass (`pytest tests/test_rubric.py`)
- All existing tests unaffected (`pytest tests/` -- same failures as before, all pre-existing)
- Tested on H200 GPU with OLMo3-7B Think SFT models (baseline vs ours) using Qwen3-30B-A3B as judge on alpaca-eval, arena-hard, and m-arena-hard-EU. Results with Likert-enforced prompt to follow.
