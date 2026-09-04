*This project has been created as part of the 42 curriculum by glieuw-a*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) pipeline over the vLLM 0.10.1 codebase.
Given a natural-language question about vLLM, the system retrieves the most
relevant spans of the repository's documentation and source code with BM25
lexical search, then feeds those spans to a small local language model
(Qwen/Qwen3-0.6B) which produces an answer grounded only in the retrieved
context.

The pipeline is written in Python and exposed as a CLI with six commands:
build the index, search a single query, search a whole question dataset,
answer a single query, answer a whole dataset, and evaluate saved search
results against ground truth with recall@k.

## System architecture

One module per pipeline stage, with typed data flowing between them:

```
data/raw/ (vLLM corpus)
      |
      v
chunking.py   corpus files -> language-aware chunks (LangChain splitters)
      |
      v
index.py      chunks -> tokenized, persisted BM25 index (bm25s) in data/processed/
      |
      v
search.py     query -> ranked MinimalSource spans (file path + character offsets)
      |
      v
answer.py     spans -> text read back from disk -> grounded prompt
      |
      v
llm.py        prompt -> Qwen3-0.6B answer (transformers, batched, greedy)
```

Supporting modules:

| Module | Role |
|---|---|
| `models.py` | 8 pydantic models (`MinimalSource`, `UnansweredQuestion`, `AnsweredQuestion`, `RagDataset`, `MinimalSearchResults`, `MinimalAnswer`, `StudentSearchResults`, `StudentSearchResultsAndAnswer`) validating every data boundary |
| `io.py` | JSON and file I/O, with pydantic validation on every load |
| `evaluation.py` | Self-check reimplementation of the grader's recall@k (IoU > 0.05, exact file-path match) |
| `errors.py` | `InputError`, the single user-facing error type |
| `__main__.py` | The Fire CLI: `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate` |

The stages only communicate through validated models: the search output a
question was answered from is exactly the `MinimalSearchResults` that was
saved to disk, and every file read back in is re-validated before use.

## Instructions

### Requirements

- Python >= 3.10
- [uv](https://docs.astral.sh/uv/)
- ~1.5 GB of disk for the Qwen3-0.6B weights (downloaded on first `answer`)

### Installation

```sh
make install        # uv sync
```

`torch` is pinned to the CPU wheel index in `pyproject.toml`: the pipeline
never touches a GPU, and the default resolution would otherwise pull several
gigabytes of CUDA runtime onto a CPU-only machine.

### Data layout

The CLI defaults assume this layout (all paths overridable by flags):

```
data/
  raw/                                   # the vLLM corpus to index
  processed/                             # the persisted BM25 index
  datasets/
    UnansweredQuestions/                 # question-only datasets
    AnsweredQuestions/                   # ground truth (sources + answers)
  output/
    search_results/<Scope>/              # saved StudentSearchResults
    search_results_and_answer/<Scope>/   # saved answers
```

### Execution

```sh
# 1. Build the index (~10 s for the full corpus)
uv run python -m src index --max_chunk_size 2000

# 2. Search a single question
uv run python -m src search "How to configure the OpenAI server?" --k 5

# 3. Search a whole dataset and save the results
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions/

# 4. Answer a single question (~40 s on a 4-core CPU)
uv run python -m src answer "How to configure the OpenAI server?" --k 10

# 5. Answer every question in saved search results
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions/

# 6. Evaluate saved search results against ground truth
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json
```

Make wrappers exist for every command (`make run-index`, `make run-search
query="..." k=5`, `make run-search-dataset`, `make run-answer`,
`make run-answer-dataset`, `make run-evaluate`), plus:

```sh
make run       # show the CLI help
make lint      # flake8 + mypy
make clean     # remove caches
```

## Example usage

Both single-query commands print the same JSON schema their batch
counterparts write, so one question can be piped through the same tooling.
`answer` prints the prose answer first, for a reader, then the document:

```
$ uv run python -m src search "How to configure the OpenAI server?" --k 3
{
  "search_results": [
    {
      "question_id": "8c437f79-a32d-43d0-bd0b-3e24b2892b6c",
      "question": "How to configure the OpenAI server?",
      "retrieved_sources": [
        {
          "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
          "first_character_index": 0,
          "last_character_index": 1952
        },
        {
          "file_path": "data/raw/vllm-0.10.1/docs/configuration/serve_args.md",
          "first_character_index": 0,
          "last_character_index": 1015
        }
      ]
    }
  ],
  "k": 3
}

$ uv run python -m src answer "How to configure the OpenAI server?" --k 3
To configure the OpenAI compatible server, you can use the `vllm serve`
command with the appropriate arguments. ...

{
  "search_results": [
    {
      "question_id": "8c437f79-a32d-43d0-bd0b-3e24b2892b6c",
      "question": "How to configure the OpenAI server?",
      "retrieved_sources": [ ... ],
      "answer": "To configure the OpenAI compatible server, you can use the
                 `vllm serve` command with the appropriate arguments. ..."
    }
  ],
  "k": 3
}
```

A query that holds no indexable token (`"???"`, only stopwords) is not an
error: it retrieves nothing and returns an empty `retrieved_sources` list.

`search_dataset` writes a `StudentSearchResults` JSON (one entry per
question, each with its top-k `retrieved_sources`); `answer_dataset` reads
that file back and writes the same structure with an `answer` field added.

## Chunking strategy

- **Language-aware recursive splitting.** `.py` files are split with
  `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)`, which
  prefers class/function boundaries; `.md` and `.txt` files use
  `Language.MARKDOWN`, which prefers header boundaries. The markdown choice
  matters directly for the grade: 79–83% of the docs datasets' reference
  spans begin at a `#` header, so header-aligned chunk starts line up with
  the exact sections questions cite.
- **2000-character maximum** (`--max_chunk_size`, never exceeded). The
  grader rejects any retrieved source longer than 2000 characters, so the
  cap is a hard correctness constraint, not just a tuning knob.
- **250-character overlap** between consecutive chunks, as insurance
  against an identifier or sentence straddling a chunk boundary and being
  matchable from neither side. Below a 1000-character chunk size the
  overlap scales down to a quarter of it, so small `--max_chunk_size`
  values stay usable.
- **Whitespace-only chunks are dropped.** Splitting keeps whitespace so the
  stored offsets stay exact, which leaves a handful of blank chunks; indexed,
  they carry only their file path and could take a top-k slot from a real
  match.
- Each chunk keeps its file path and its exact start offset in the file
  (`add_start_index=True`), so retrieval can report precise
  `[first_character_index:last_character_index]` spans.

## Retrieval method

Lexical retrieval with **BM25** (the `bm25s` library, k1=1.5, b=0.75).
BM25 scores a chunk by summing, over query terms, the term's inverse
document frequency weighted by a saturated term frequency normalised for
chunk length — rare shared terms count a lot, common ones little. Ranking
is a straight sort on that score; the top-k spans are returned (k clamped
to the corpus size).

What makes it work on a codebase is everything around the scoring:

- **Identifier-aware tokenizer, shared between index and query time.**
  Compound identifiers (`snake_case`, `CamelCase`, digit-suffixed) emit
  both the whole identifier and every subword. `bm25s`'s default tokenizer
  lowercases before splitting, so `AsyncLLMEngine` becomes one opaque token
  that a question phrased "async LLM engine" can never match; emitting the
  subwords too makes both phrasings hit.
- **Stemming** (PyStemmer, English) on every token, aligning
  "scheduling"/"scheduler" and "cached"/"caching" across query and corpus.
- **Extended stopwords.** The standard English list plus interrogatives
  ("what", "how", "does", ...). Every query is a question but the corpus
  is not, so question words carry no discriminative signal and only add
  noise to the score.
- **File-path prefix in the indexed text.** Each chunk is indexed as its
  file path plus its body — the stored spans are untouched. The path names
  the file's topic (`docs/cli/serve.md`), which queries often mention while
  a short chunk body may not. This is the single largest contributor to the
  score (see the ablation below).

## Performance analysis

Recall@5 under the official grader (IoU > 0.05 with exact file-path match):

| Dataset | Public | Private | Threshold |
|---|---|---|---|
| docs | **0.880** | **0.820** | >= 0.80 |
| code | **0.830** | **0.770** | >= 0.50 |

Full recall curves (R@1 / R@3 / R@5 / R@10):

| Dataset | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| docs public | 0.68 | 0.84 | 0.88 | 0.94 |
| docs private | 0.65 | 0.80 | 0.82 | 0.84 |
| code public | 0.56 | 0.77 | 0.83 | 0.89 |
| code private | 0.57 | 0.74 | 0.77 | 0.83 |

The code margin is wide (0.77 vs a 0.50 threshold). The docs margin is
thin — 0.820 vs 0.80 is roughly two questions on a 100-question set —
which is why every retrieval decision below was measured rather than
assumed.

**Speed**: indexing the full corpus (1843 indexed files -> 14,710 chunks) takes
~10 s (limit 300 s); retrieving 200 questions takes ~9 s (limit 90 s);
a single `answer --k 10` takes ~40 s on a 4-core CPU.

### Ablation: why each decision is there

Recall@5 with one decision disabled at a time:

| Config | docs private | code private | docs public | code public |
|---|---|---|---|---|
| **shipped** | **82.0** | **77.0** | **88.0** | **82.8** |
| no file-path prefix | 79.0 | 68.0 | 85.0 | 73.7 |
| base stopwords (no interrogatives) | 82.0 | 77.0 | — | — |
| no stemming | 85.0 | 75.0 | 55.0 | 61.6 |
| no identifier subwords | 83.0 | 69.0 | — | — |
| no chunk overlap | 83.0 | 77.0 | — | — |

Read honestly: the file-path prefix and identifier subwords are clear wins,
especially on code. Stemming is a trade-off — it *costs* 3 points on the
private docs set while being worth +33 on the public one; it stays because
it helps three of the four sets and the private docs score still clears the
threshold with it. The extended stopword list does not move Recall@5 at all
— it earns its place by lifting Recall@1 by 2–3 points. Overlap is roughly
neutral on Recall@5 and is kept as cheap insurance against boundary splits.

## Design decisions

- **One module per stage, pydantic at every boundary.** Every JSON file
  read or written goes through one of the 8 models in `models.py`, so a
  malformed dataset or a hand-edited results file fails loudly at load
  time, not deep inside a batch of 200 questions.
- **The error-handling contract.** Exceptions are raised low, with a
  message naming the offending value; foreign exceptions are wrapped
  (`raise InputError(f"cannot read {path}: {error}") from error`) so the
  original cause is preserved; a frame catches only when it has a genuine
  local recovery — e.g. `answer.py` skips one unreadable source out of k
  and carries on; and there is exactly one `except InputError`, in
  `main()`, which prints to stderr and returns exit code 1. The
  alternative — returning `None` or error strings — was rejected because
  it forces every caller to re-check and lets errors flow onward as if
  they were data.
- **BM25 parameters left at library defaults** (k1=1.5, b=0.75). Tuning
  them was measured and gave nothing once chunk overlap was in place, so
  the defaults were kept rather than shipping an unjustifiable magic
  number.
- **Answer generation kept honest and reproducible.** Qwen3-0.6B via
  transformers, chat template applied with `enable_thinking=False`
  (otherwise the model spends its whole token budget inside a `<think>`
  block), greedy decoding (`do_sample=False`) for reproducible answers,
  `max_new_tokens=192`, left-padded batches of 4, and the prompt sliced
  off the generated ids before decoding.
- **Context capped at 6000 characters, filled by retrieval rank.** The
  full top-k still goes into the JSON output — only what reaches the model
  is bounded. On CPU, prefill cost scales with prompt length, and this cap
  took a single `answer --k 10` from ~15 minutes to ~40 seconds.
- **The prompt instructs the model to answer only from the numbered
  context** and to say so when the context is insufficient, rather than
  letting a 0.6B model improvise.

## Challenges faced

- **The model echoing the prompt back as the answer.** `generate()` returns
  prompt + completion ids; decoding them all made every "answer" start with
  the question and context. Fixed by slicing the input length off the
  generated ids before decoding.
- **Qwen3's thinking mode eating the whole budget.** With thinking enabled,
  the model spent all 192 new tokens inside a `<think>` block and produced
  no visible answer. Fixed with `enable_thinking=False` in the chat
  template (plus a stripper for any `<think>` span that still appears).
- **CPU prefill speed.** Long contexts made a single answer take ~15
  minutes; the 6000-character context budget brought it to ~40 seconds
  without changing what gets saved to disk.
- **A real overfitting episode.** A first tuning round reached 0.870 on
  public docs but only 0.760 on private docs — below threshold. The fix
  was methodological: treat the public set as training data and the
  private set as held-out validation, and reject any change that helps one
  while hurting the other.
- **An "obviously right" idea that measured worse.** Capping chunks per file
  to diversify the top-k seemed like a free win, but scored 8–11 points
  *worse*: consecutive overlapping chunks of the right file cover different
  regions, and each one is an independent chance to clear the IoU bar.
  Diversification traded those chances away for hits in the wrong files.

## Resources

### References

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and
  Beyond* (2009) — the BM25 ranking function used here.
- Sparck Jones, *A statistical interpretation of term specificity and its
  application in retrieval* (1972) — the origin of IDF / TF-IDF weighting.
- Lewis et al., *Retrieval-Augmented Generation for Knowledge-Intensive
  NLP Tasks* (2020) — the RAG pattern this project implements.
- [bm25s documentation](https://github.com/xhluca/bm25s) — the BM25
  implementation.
- [LangChain text splitters](https://python.langchain.com/docs/concepts/text_splitters/)
  — the recursive language-aware chunking.
- [Qwen3 model card](https://huggingface.co/Qwen/Qwen3-0.6B) — chat
  template and thinking-mode behaviour.

### AI use

AI assistance was used for code review and design advice, not for writing the
implementation:

- **Code review** — reading the code for defects and reporting them with
  file and line references, which I then fixed myself: the chunk extractor
  losing the relative path (`file.name` instead of the path the grader
  compares verbatim), a `return` where a `continue` belonged that aborted the
  walk on the first non-source file, and a module-level call that walked the
  corpus on import.
- **Design advice** — discussing the error-handling contract described under
  Design decisions above: whether functions should return `None` or raise, and
  where the boundary between raising low, wrapping foreign exceptions, and
  catching once should sit.

The retrieval design itself (chunking, tokenizer, stopwords, stemming, path
prefix) was decided by measurement: every choice in this README is backed by
the numbers in the ablation table, including the ones where the measurement
said "no" (per-file diversification, BM25 parameter tuning).
