# `llm-citation-erasure-bench`

![Python 3.9+ | 0 Deps](https://img.shields.io/badge/Python_3.9%2B-0_Dependencies-111827) ![License: MIT](https://img.shields.io/badge/License-MIT-059669)

**25 Hand-Written Illustrative Test Cases for LLM Note-Quote Grounding and Rolling-Summary Note Deletion**

| Naive Check Tested | Synthetic User Note / Action | Illustrative Failure Mode |
| :--- | :--- | :--- |
| `quote.lower() in weekly_notes.lower()` | `"the architecture draft remained unshared"` | Substring check matches `"shared"` inside `"unshared"` |
| `quote.lower() in weekly_notes.lower()` | `"I didn't submit the report before the deadline"` | Substring check matches `"submit the report"` after negation |
| `DELETE FROM notes WHERE id = 'note_3'` (stale summary kept) | User deletes `"took 40mg propranolol before board pitch"` | Cached weekly summary still retains `"managed with beta-blocker medication"` |

---

## 1. Motivation & Exploratory Origin

This repository packages **25 hand-written illustrative cases** (`15` quote-grounding traps in Suite A and `10` five-day note-deletion logs in Suite B) into a zero-dependency Python 3 script (`run_benchmark.py`) to test simple baseline guardrails and local Ollama models.

- **Exploratory Background Note:** Suite A was motivated by one small exploratory test I wrote and scored on 4 fictional weekly-habit cases (`2026-09-28`): Apple's on-device Foundation Model echoed phrases from the user's goal text in `20 of 63` quotes (`4 of 20` runs right, rating an empty-notes case `10/10` five times out of five), whereas Claude Sonnet (`claude` CLI) produced `0 of 23` goal-echo quotes (`8 of 8` runs right). Raw outputs from that exploratory check were not archived; the 25 JSON fixtures in [`suites/synthetic_failure_cases.json`](./suites/synthetic_failure_cases.json) were written by hand afterward to make specific edge cases reproducible.

### Suite A: Quote-Span & Polarity Edge Cases (`15` Synthetic Cases)
- **Baseline A1 (`quote in full_prompt`):** Fails `15/15` synthetic cases because quotes copied from the Goal Rubric on an empty log still appear inside the prompt.
- **Baseline A2 (`quote.lower() in weekly_notes.lower()`):** Blocks the `4/4` empty-log rubric echoes, while passing `11/15` cases where the substring sits inside an antonym (`"shared"` in `"unshared"`), after a negation (`"didn't submit the report"`), or inside an unfulfilled wish (`"plan to run 5 miles tomorrow"`).

### Suite B: Cached Summary Tradeoffs After Deleting One Note (`10` Synthetic 5-Day Logs)
When a user deletes `note_3` from a 5-day log:
- **Baseline B1 (delete row, keep stale cached summary):** `0` extra LLM calls, `10/10` retain paraphrased details from deleted `note_3` (`"40mg propranolol"` survives as `"beta-blocker medication"`).
- **Baseline B2 (naive per-note map loop over surviving notes):** Clears deleted markers (`0/10` leaks) at the cost of `N-1` (`4` per case) LLM calls.
- **Baseline B3 (drop cached summary on delete; regenerate lazily only when next viewed):** Clears deleted markers immediately (`0/10` leaks) with `0` immediate LLM calls at deletion time.
- **Baseline B4 (invalidate cache and run `1` batch re-summarization call over surviving notes):** Clears deleted markers (`0/10` leaks) in `1` LLM call per deletion (`10` calls across `10` cases).

---

## 2. Quickstart (`0` Dependencies · Python 3.9+ Standard Library Only)

### Mode 1: Offline Baseline Self-Test (`--demo-baseline`, `0` LLM Calls)

```bash
python3 run_benchmark.py --demo-baseline
```

### Mode 2: Local Ollama Model Evaluation (`http://localhost:11434`)

`run_benchmark.py` restricts HTTP requests to `localhost` / `127.0.0.1` / `::1`:

```bash
ollama pull llama3.2:3b
python3 run_benchmark.py --ollama-model llama3.2:3b
```

---

## 3. Repository Files & Notices

- [`run_benchmark.py`](./run_benchmark.py): Standard-library Python 3 runner (`--demo-baseline` / `--selftest` and `--ollama-model`).
- [`suites/synthetic_failure_cases.json`](./suites/synthetic_failure_cases.json): `25` hand-written synthetic fixtures (`A01`-`A15` and `B01`-`B10`).
- [`LICENSE`](./LICENSE) & [`PATENTS.md`](./PATENTS.md): Standard MIT copyright license on the scripts and synthetic examples, with patent rights reserved (`U.S. Provisional Patent Appl. No. 64/204,846`; see [`PATENTS.md`](./PATENTS.md)).
