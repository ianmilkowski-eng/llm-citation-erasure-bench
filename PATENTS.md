# Explicit Reservation of Patent Rights & No Patent License Notice

**Author & Patent Rights Holder:** Ian Milkowski (Milkplexity Labs)  
**Inventor Pedigree:** Sole Inventor, `U.S. Provisional Patent Application No. 64/204,846` (*U.S. Patent Pending*, covering Milkplexity Labs' proprietary Ciało remediation and state-verification architecture)

---

## 1. Synthetic Failure-Mode Benchmark Scope Only (Zero Remediation Code)

This repository (`llm-citation-erasure-bench`) distributes **strictly synthetic benchmark prompts, hand-labelled guardrail unit-test fixtures, and baseline diagnostic harness scripts** designed to reproduce failure modes in standard open-source structured-JSON decoding and retrieval-augmented generation (`RAG`) / rolling-memory pipelines.

It contains **zero** proprietary remediation logic, production verification gates, state-ledger engines, or runtime source code from Milkplexity Labs or Ciało, and **this synthetic benchmark repository is not itself a patented or patent-pending article** (`35 U.S.C. § 292`).

## 2. Express Non-Grant & Reservation of Patent Rights

The [MIT License](./LICENSE) accompanying this repository applies **solely to copyright** in the synthetic benchmark harness scripts ([`run_benchmark.py`](./run_benchmark.py)) and synthetic test vectors ([`suites/synthetic_failure_cases.json`](./suites/synthetic_failure_cases.json)) distributed herein.

Notwithstanding any copyright permission in the [MIT License](./LICENSE), **no patent license, immunity, authorization, covenant not to sue, or patent exhaustion right (whether express, implied, by equitable estoppel, or otherwise) is granted** under `U.S. Provisional Patent Application No. 64/204,846` or under any U.S. provisional application, non-provisional utility application, continuation, divisional, international (`PCT`) application, foreign counterpart application, or issued patent owned, filed, or controlled by **Ian Milkowski**. All patent rights are expressly reserved in full by Ian Milkowski.

## 3. Local Open-Weight Execution Scope Only

The benchmark runner in this repository executes deterministically offline (`--demo-baseline`, alias `--selftest`) to unit-test naive Python guardrails with `0` LLM calls, or against a user-operated local loopback open-weight model endpoint (`http://localhost:11434/api/generate` via `--ollama-model`). It is not designed or authorized to probe, load-test, or evaluate third-party commercial SaaS APIs or hosted vendor services.
