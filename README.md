# CLAPnq Full-Gold vs Closed-book

A reproducible evaluation pipeline for one focused question:

> Does providing an answer-bearing Gold passage improve a model's answer correctness over closed-book answering?

The project evaluates the same CLAPnq answerable questions under two conditions:

- `gold`: system instruction + title + full Gold passage + question;
- `closed_book`: the same system instruction + question only.

The correctness Judge receives only the question, all valid human reference answers, and the candidate answer. It never receives the passage, selected sentences, generator identity, or experimental condition.

## Project layout

```text
clapnq_eval/
├── configs/experiment.yaml       # Frozen experiment and sampling parameters
├── scripts/
│   ├── serve_generator.sh        # vLLM server for one 7B/8B generator
│   ├── serve_judge.sh            # SGLang server for Qwen3.8-27B-FP8
│   └── run_tests.sh
├── src/clapnq_eval/
│   ├── prompts.py                # Frozen generation and Judge prompts
│   ├── schema.py                 # Judge JSON Schema (label, then reason)
│   ├── validate.py               # Shared generation/judgment resume checks
│   ├── data.py                   # Download, checksum, and normalization
│   ├── generate.py               # Concurrent generation + signed resume
│   ├── judge.py                  # JSON-Schema correctness Judge
│   ├── metrics.py                # EM, token-F1, ROUGE, optional BERTScore
│   ├── report.py                 # Coverage checks, bootstrap, McNemar
│   ├── manifest.py               # Frozen run signature and prompt snapshots
│   └── cli.py                    # Command-line interface
├── tests/
├── data/                         # Ignored downloaded/normalized data
└── runs/                         # Ignored experiment outputs
```

## Local environments

The server scripts use the existing environments and model paths:

| Role | Environment | Backend/model |
|---|---|---|
| Generation | `ACL2027-vllm` | vLLM 0.27.1; Qwen2.5-7B, Llama-3.1-8B, Mistral-7B |
| Orchestration/Judge | `ACL2027-sglang` | SGLang 0.5.17; Qwen3.8-27B-FP8 |

Install the lightweight client package in the SGLang environment:

```bash
cd /home/qluai/ZJS/ACL2027/clapnq_eval
conda activate ACL2027-sglang
pip install -e .
```

For optional BERTScore:

```bash
pip install -e '.[semantic]'
```

BERTScore is disabled by default because it downloads an additional encoder.

## Frozen prompts

The common generation instruction is a system message. Gold and closed-book use the same instruction and differ only in the user-provided evidence. Each generator uses its native chat template.

The Judge is reference-based and pointwise. It returns one structured label:

- `CORRECT`;
- `MINOR_ERROR`;
- `MAJOR_ERROR`.

`CORRECT` alone counts toward strict Judge accuracy. `CORRECT + MINOR_ERROR` forms the non-major rate.

At runtime, exact prompt snapshots, SHA-256 fingerprints, and a frozen run
signature are written to:

```text
runs/<run-name>/prompts/
runs/<run-name>/manifest.json
```

Changing the data, seed, prompts, model names, generation settings, or Judge
settings under an existing signed run is rejected before inference. Use
`--run-name <new-name>` for a new experiment.

## Judge parameters

The default configuration follows the official Qwen3.8 non-thinking profile
bundled with the local model, with one structured-output exception:

```text
temperature       = 0.7
top_p             = 0.8
top_k             = 20
min_p             = 0.0
presence_penalty  = 0.0   # official chat non-thinking is 1.5
repetition_penalty= 1.0
enable_thinking   = false
```

The JSON Schema decodes `label` first, then `reason`. Official chat
`presence_penalty=1.5` is for open-ended generation; it is the wrong prior
for a three-way classification label, so this Judge uses `0.0`. All other
non-thinking sampling values match the official README.

The source is the model's local official README and generation configuration:

```text
/home/qluai/.cache/modelscope/hub/models/Qwen/Qwen3___8-27B-FP8/README.md
/home/qluai/.cache/modelscope/hub/models/Qwen/Qwen3___8-27B-FP8/generation_config.json
```

The corresponding official Hugging Face checkpoint metadata is frozen as:

```text
revision     = 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
lastModified = 2026-08-14
```

The checkpoint's `generation_config.json` describes its thinking-enabled
default. This evaluation deliberately uses non-thinking inference: every
SGLang Judge request sets
`extra_body.chat_template_kwargs.enable_thinking=false`. Thinking mode is
the model's strongest official profile and should be a separately
calibrated ablation with a much larger `max_tokens`.

A fixed request seed and server seed are used. The SGLang script also supports batch-invariant deterministic inference for calibration:

```bash
DETERMINISTIC_INFERENCE=1 bash scripts/serve_judge.sh
```

This option can reduce throughput. A greedy Judge profile (`temperature=0`) should be treated as a separately calibrated ablation, not as the official Qwen3.8 profile.

The Judge server uses TP=2, text-only model loading, CPU transport for unused
multimodal features, and five concurrent requests. The five-request default
matches the Mamba state-cache limit observed for this checkpoint on the two
local RTX 4090 D GPUs.

## 1. Prepare CLAPnq

The configured file is the official public answerable development split,
pinned to Hugging Face revision
`b7f27c581c2cbe6e6affa121bb4194ae08fc6133`. Its expected SHA-256 is also
frozen in `configs/experiment.yaml`.

```bash
PYTHONPATH=src python -m clapnq_eval prepare
```

The loader:

- preserves string IDs, including negative integer-like IDs;
- uses `passages[0].text` as the Full-Gold passage;
- excludes `meta.skip=true`, empty, `NA`, and `N/A` outputs;
- preserves and deduplicates all remaining human references;
- retains `selected_sentences` for audit but never sends them to generators or the Judge.

## 2. Generate answers

Start one generator on one GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_generator.sh qwen2.5-7b
```

In another terminal, generate both conditions:

```bash
PYTHONPATH=src python -m clapnq_eval generate --model qwen2.5-7b
```

Repeat after restarting the server:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/serve_generator.sh llama3.1-8b
PYTHONPATH=src python -m clapnq_eval generate --model llama3.1-8b

CUDA_VISIBLE_DEVICES=0 bash scripts/serve_generator.sh mistral-7b
PYTHONPATH=src python -m clapnq_eval generate --model mistral-7b
```

For a smoke test, isolate outputs with a separate run name:

```bash
PYTHONPATH=src python -m clapnq_eval --run-name smoke \
  generate --model qwen2.5-7b --limit 5
```

A later full invocation under `main` is independent of `smoke`. vLLM is launched with `--generation-config vllm`, while request parameters explicitly freeze greedy generation:

```text
temperature=0, top_p=1, top_k=0, min_p=0,
presence_penalty=0, repetition_penalty=1, max_tokens=256
```

## 3. Judge answers

After generation is complete, stop the generator and start the two-GPU Judge:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/serve_judge.sh
```

Judge all available outputs:

```bash
PYTHONPATH=src python -m clapnq_eval judge --model all
```

The request uses SGLang JSON Schema/XGrammar. Every response is parsed again
with Pydantic. Connection, timeout, rate-limit, and retryable server failures
use exponential backoff with jitter. Invalid schemas, invalid JSON, empty
content, and truncation are treated as per-example failures: they are written
to `runs/<run-name>/judgments/failed/<model>.<condition>.jsonl` and the rest
of the file continues. A later `judge` invocation retries only IDs that are
still missing from the success file. Infrastructure failures after retries
still abort the process.

The JSONL success output stores the label, short reason, raw structured
response, request parameters, usage, latency, prompt hash, source-example
hash, generation-row hash, and timestamp. `status` reports both success
counts and quarantined failure counts.

## 4. Compute metrics

```bash
PYTHONPATH=src python -m clapnq_eval score
```

Outputs:

```text
runs/main/metrics/per_example.jsonl
runs/main/metrics/coverage.json
runs/main/metrics/summary.json
runs/main/metrics/summary.csv
runs/main/metrics/paired.json
runs/main/metrics/paired.csv
```

Condition-level metrics include:

- strict Judge accuracy;
- non-major rate;
- label counts;
- normalized EM and token-F1;
- ROUGE-1 recall/F1 and ROUGE-L F1, taking the maximum over references;
- candidate length and candidate/reference length ratio;
- optional BERTScore-F1.

Paired metrics include:

- `strict_context_gain = Acc(gold) - Acc(closed_book)`;
- strict rescue and harm rates;
- the complete `3 x 3` Closed-book-to-Gold label transition matrix;
- paired-bootstrap confidence intervals;
- exact McNemar p-values;
- token-F1 and ROUGE context gains.

Formal `score` requires exactly one judgment for every prepared example under
all conditions listed in `generation.conditions`. Missing files, missing IDs,
duplicate IDs, extra IDs, stale references, or mismatched generation hashes
cause an error instead of silently shrinking to an intersection.

`--condition all` follows that same YAML list. Condition-level metrics are
emitted for every configured condition. The gold-versus-closed-book paired
table is written only when both of those conditions are present and complete.

Incomplete smoke runs can be inspected with `status`, but `score` intentionally
has no partial-data override.

`--allow-rouge-fallback` is only for environments where `rouge-score` has not
yet been installed. Formal runs fail fast rather than silently changing the
ROUGE implementation.

Check progress at any time:

```bash
PYTHONPATH=src python -m clapnq_eval status
```

## Resume and reproducibility

Generation and Judge outputs are append-only during execution and keyed by
`example_id`. Each output file has a non-blocking single-writer lock. If a
process stops during a JSONL write, the unterminated tail is removed before the
next append.

Resume validates the source-example hash, prompt version/hash, served model,
candidate answer, all decoding parameters, and Judge input hash before skipping
an existing ID. The run-level manifest independently freezes the data, seed,
prompts, models, generation settings, and Judge settings.

Use `--run-name <name>` or change `run.name` in the YAML whenever starting a
new configuration. Incompatible settings cannot be mixed in one signed run.

## Tests

```bash
bash scripts/run_tests.sh
```

The test suite is fully offline. It covers CLAPnq reference cleaning, prompt
separation, JSON Schema labels, crash-safe JSONL resume, run signatures,
non-retryable validation failures, formal input completeness, lexical metrics,
the 3 x 3 label transition matrix, bootstrap intervals, and McNemar
calculations.

## Interpretation boundary

With only Full-Gold and closed-book conditions, the supported claim is:

> Providing an answer-bearing Gold passage changes or improves reference-based answer correctness relative to closed-book answering.

This design does not by itself prove that a particular selected evidence sentence caused the answer, and the correctness Judge is intentionally not a passage-faithfulness evaluator.
