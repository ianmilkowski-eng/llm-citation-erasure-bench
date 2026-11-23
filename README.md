# `llm-citation-erasure-bench`

![Python 3.9+ | 0 Deps](https://img.shields.io/badge/Python_3.9%2B-0_Dependencies-111827) ![Harness License: MIT (Copyright Only)](https://img.shields.io/badge/Harness_License-MIT_(Copyright_Only)-059669) ![Research: Milkplexity Labs](https://img.shields.io/badge/Research-Milkplexity_Labs-2563EB)

**Why `100%` Schema-Valid JSON Citations and `DELETE FROM notes` Vector Purges Still Invert Ground Truth and Leak Deleted GDPR Art. 17 Memories**

| What Passed Standard CI Guardrails (`Green`) | User Input / Action | What Actually Shipped to Production (`Red`) |
| :--- | :--- | :--- |
| `quote.lower() in weekly_notes.lower() == True` | `"the architecture draft remained unshared"` | Model cites `"shared"` inside `"unshared"` and scores **`8/10`** |
| `quote.lower() in weekly_notes.lower() == True` | `"I didn't submit the report before the deadline"` | Model cites `"submit the report"` and scores **`9/10`** |
| `DELETE FROM notes WHERE id = 'note_3'` (`0` rows left) | User deletes `"took 40mg propranolol before board pitch"` (GDPR Art. 17) | Rolling weekly summary cache still states `"managed with beta-blocker medication"` |

---

## 1. Two Silent Bugs in Standard Open-Source LLM & RAG Pipelines

Teams shipping wearable coaching, health journals, personal knowledge vaults, and enterprise RAG assistants routinely rely on two standard patterns that pass syntactic unit tests while failing silently on longitudinal user logs:

### Failure Class A: Schema-Constrained JSON Goal-Echo, Negation, Wish, & Torn-Stem Citations
- **Observed Exploratory `~3B` Open-Weight Baseline:** When grammar-constrained decoding (`GBNF` / JSON Schema with `minItems: 1` on `evidence_quotes`) is run on empty or sparse weekly logs using local `~3B` models, **`20/63` (`31.7%`)** of returned citations copy target phrases directly out of the **system prompt's Goal Rubric**, scoring empty weeks `9/10` or `10/10`.
- **Why `quote.lower() in notes.lower()` Still Fails:** Checking that every cited quote appears in the user's raw notes blocks empty-log rubric echoes, **but still passes `100%` of polarity-inverted and mid-word substring slices**:
  - **Negated Clauses:** `"I didn't submit the report"` -> cites `"submit the report"` (`True` substring match).
  - **Unfulfilled Future Wishes:** `"I plan to run 5 miles tomorrow"` -> cites `"run 5 miles"` (`True` substring match).
  - **Torn Antonym Stems:** `"remained unshared"` / `"severely dehydrated"` -> cites `"shared"` / `"hydrated"` (`True` substring match).

### Failure Class B: The GDPR Art. 17 Rolling-Memory Erasure Dilemma
When a user logs `5` daily notes (`Note #1`..`Note #5`) and issues a **GDPR Art. 17 Right to Erasure** deletion on `Note #3` (for example, a medication dose, fertility clinic lab, therapy session, or stolen SSH key):
- **Trap (i) Paraphrased Ghost Residue:** Deleting `Note #3` from SQLite/Postgres and `pgvector` leaves **semantic paraphrases** inside cached rolling summaries (`"40mg propranolol"` survives as `"beta-blocker medication"`, `"Zurich"` survives as `"Switzerland"`). Exact-string stripping misses every paraphrase.
- **Trap (ii) `O(N)` Re-Summarization Amplification:** Nuking cached summaries on every single-note deletion purges the ghost echo only by forcing **`N-1` redundant LLM inference calls** across untouched surviving notes, draining mobile NPU/battery budgets or multiplying cloud token bills whenever a user edits one line.

---

## 2. Quickstart (`0` Dependencies · Python 3.9+ Standard Library Only)

[`run_benchmark.py`](./run_benchmark.py) uses only the Python 3 standard library and enforces a strict loopback guard (`localhost` / `127.0.0.1` / `::1`) so it never calls commercial SaaS APIs.

### Mode 1: Deterministic Guardrail Unit-Test (`--demo-baseline`, `0` LLM Calls, `<50ms`)

Feeds `25` synthetic adversarial fixture payloads directly into standard OSS Python verifiers (`quote in prompt`, `quote.lower() in notes.lower()`, and SQL row-delete + rolling summary cache) with **zero LLM calls** to verify deterministically how naive guardrails behave:

```bash
python3 run_benchmark.py --demo-baseline
```

*(Note: `--selftest` is retained as a backwards-compatible CLI alias.)*

### Mode 2: Live Local Open-Weight Model Evaluation via Ollama (`llama3.2:3b`, `qwen2.5:3b`, etc.)

Passes structured JSON Schema decoding (`minItems: 1` on `evidence_quotes`) and 5-day rolling-summary prompts to any local open-weight model running on [Ollama](https://ollama.com) (`http://localhost:11434`):

```bash
# Example 1: Local 3B Llama 3.2
ollama pull llama3.2:3b
python3 run_benchmark.py --ollama-model llama3.2:3b

# Example 2: Local 3B Qwen 2.5 (save machine-readable JSON output)
ollama pull qwen2.5:3b
python3 run_benchmark.py --ollama-model qwen2.5:3b --json-out local_qwen25_3b_results.json
```

---

## 3. Deterministic Guardrail Verification (`--demo-baseline`, `0` LLM Calls)

Running `python3 run_benchmark.py --demo-baseline` unit-tests standard naive Python guardrails against the `25` adversarial fixtures in [`suites/synthetic_failure_cases.json`](./suites/synthetic_failure_cases.json):

| Suite & Standard Naive OSS Guardrail | Cases | Guardrail Failures | Failure Rate | Verified Guardrail Blind Spot |
| :--- | :---: | :---: | :---: | :--- |
| **Suite A · Baseline A1** (`JSON Schema + Prompt Substring`) | `15` | **`15/15`** | **`100.0%`** | Accepts rubric goal-echoes (`10/10` on empty logs), negations, future wishes, & torn stems |
| **Suite A · Baseline A2** (`quote.lower() in notes.lower()`) | `15` | **`11/15`** | **`73.3%`** | Blocks empty-log echoes (`4/4`), but passes `11/11` negations, future wishes, & mid-word stems |
| **Suite B · Baseline B1** (`DELETE Note #3` + Cached Summary) | `10` | **`10/10`** | **`100.0%`** | **GDPR Art. 17 Residue:** Paraphrased ghost echoes of deleted `Note #3` survive in memory |
| **Suite B · Baseline B2** (`Invalidate Cache + N-1 Re-Poll`) | `10` | **`10/10`** | **`100.0%`** | **`O(N)` Amplification:** Forces `40` redundant LLM calls across `40` untouched surviving notes |

---

## 4. Repository Layout

- [`README.md`](./README.md): Problem overview, `31.7%` (`20/63`) exploratory `~3B` goal-echo observation, and reproduction commands.
- [`run_benchmark.py`](./run_benchmark.py): Zero-dependency Python 3 CLI (`--demo-baseline` offline guardrail unit-test + loopback-only `--ollama-model <model>`).
- [`suites/synthetic_failure_cases.json`](./suites/synthetic_failure_cases.json): `25` 100% synthetic test vectors (`A01`-`A15` JSON citation/polarity traps + `B01`-`B10` 5-day GDPR Art. 17 single-note erasure logs).
- [`LICENSE`](./LICENSE): Standard MIT copyright license for the synthetic benchmark scripts and fixtures only, subject to [`PATENTS.md`](./PATENTS.md).
- [`PATENTS.md`](./PATENTS.md): Explicit reservation of patent rights and non-grant notice.

---

## 5. Author & Patent Reservation Notice

Built by **Ian Milkowski** ([Milkplexity Labs](https://ianmilkowski.com)), Sole Inventor of **`U.S. Provisional Patent Application No. 64/204,846`** (*U.S. Patent Pending*, covering Milkplexity Labs' proprietary Ciało local-first state and grounding remediation architecture).

This repository distributes **synthetic failure-mode test vectors and baseline diagnostic scripts only** (containing zero proprietary remediation code and **not itself a patented or patent-pending article**). Per [`PATENTS.md`](./PATENTS.md), the [MIT License](./LICENSE) applies **solely to copyright** in these benchmark scripts and synthetic fixtures and grants **zero express or implied patent license, covenant not to sue, or patent exhaustion** under any patent application or issued patent of Ian Milkowski.
