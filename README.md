*This project has been created as part of the 42 curriculum by glieuw-a*

# RAG against the machine

## Description

A Retrieval-Augmented Generation (RAG) pipeline over the vLLM 0.10.1 codebase.
Given a natural-language question about vLLM, the system retrieves the most
relevant spans of the repository's documentation and source code with BM25
lexical search, then feeds those spans to a small local language model
(Qwen/Qwen3-0.6B) which produces an answer grounded only in the retrieved
context.

Why retrieval at all: a language model only knows what was in its training
data, frozen at training time, and a 0.6B model knows little about vLLM's
internals. Retrieval hands it the relevant documentation and code at
question time, so the answer rests on the actual source, without
retraining. It also lets every answer point to the spans it was built from.

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

# 4. Answer a single question (~13 s on a 4-core CPU)
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
make lint      # flake8 . + mypy . (data/ and .venv excluded via .flake8 / pyproject)
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
          "file_path": "data/raw/vllm-0.10.1/tests/v1/logits_processors/test_custom_online.py",
          "first_character_index": 2603,
          "last_character_index": 4567
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
To configure the OpenAI server, you can use the `vllm serve` command with
the following parameters:

1. Set the model name using the `--model` argument. ...

{
  "search_results": [
    {
      "question_id": "8c437f79-a32d-43d0-bd0b-3e24b2892b6c",
      "question": "How to configure the OpenAI server?",
      "retrieved_sources": [ ... ],
      "answer": "To configure the OpenAI server, you can use the `vllm serve`
                 command with the following parameters: ..."
    }
  ],
  "k": 3
}
```

A query that holds no indexable token (`"???"`, only stopwords), or only
words that occur nowhere in the corpus (`"asdfghjkl"`), is not an error: it
retrieves nothing and returns an empty `retrieved_sources` list. BM25 would
otherwise fill the top-k with zero-score chunks, so any hit that shares no
term with the query is dropped. An empty query or `--k 0` is rejected with
an error message and exit code 1.

`search_dataset` writes a `StudentSearchResults` JSON (one entry per
question, each with its top-k `retrieved_sources`); `answer_dataset` reads
that file back and writes the same structure with an `answer` field added.

## Chunking strategy

- **Language-aware recursive splitting.** `.py` files are split with
  `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)`, which
  splits first at top-level `class` and `def` lines, then at blank lines
  (indented methods are separated by those blank lines); `.md` and `.txt`
  files use
  `Language.MARKDOWN`, which prefers header boundaries. The markdown choice
  matters directly for the grade: 79–83% of the docs datasets' reference
  spans begin at a `#` header, so header-aligned chunk starts line up with
  the exact sections questions cite.
- **2000-character maximum** (`--max_chunk_size`, never exceeded). The
  grader rejects any retrieved source longer than 2000 characters, so the
  cap is a hard correctness constraint, not just a tuning knob. `index`
  clamps a larger value back to 2000 with a warning.
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

**Why chunk size matters.** Chunks that are too small lose context: a
question's terms get spread over several chunks, so no single chunk
shares enough of them to rank high. The grader also needs span IoU > 0.05
with the reference span, and a tiny chunk can sit inside a long reference
without covering 5% of it. Chunks that are too large dilute the score:
BM25's length normalisation (`b`) penalises them, a rare matching term
counts for less among many other words, and the IoU against a short
reference span drops below the bar even when the chunk contains it. Large
chunks also make the LLM prompt longer and slower. 2000 characters is the
grader's ceiling; the sweep below measures what smaller sizes do.

## Retrieval method

Lexical retrieval with **BM25** (the `bm25s` library, k1=1.5, b=0.75).
BM25 scores a chunk by summing, over query terms, the term's inverse
document frequency weighted by a saturated term frequency normalised for
chunk length — rare shared terms count a lot, common ones little. Ranking
is a straight sort on that score; the top-k spans are returned (k clamped
to the corpus size). Chunks with a score of 0 share no term with the query
and are never returned.

**Why BM25 rather than TF-IDF.** Both weight a term by its IDF, so a rare
identifier outweighs a common word. They differ in how they treat the term
frequency (TF) inside a chunk:

- TF-IDF grows linearly (or logarithmically) with TF, so a chunk that
  repeats a word many times wins on repetition alone. Raw TF-IDF has no
  length term at all; the common cosine-normalised variant divides by the
  vector length, a fixed and full normalisation.
- BM25 saturates TF (`k1`): the tenth occurrence of a term adds far less
  than the first. Its length normalisation is partial and tunable (`b`,
  from 0 for none to 1 for full), so a long chunk does not win just
  because it holds more words, but is not punished for it either.

On code both properties matter: a long file repeats `self` and `config`
everywhere, and a short docs section should still win when it holds the
rare term the question names.

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
- **File path in the indexed text.** Each chunk is indexed as its path
  inside the repository (`docs/cli/serve.md`), written twice, followed by
  its body. The stored spans are untouched. The path names the file's
  topic, which queries often mention while a short chunk body may not.
  Writing it twice gives those topic words about the weight of a term the
  body repeats. The corpus root every file shares (`data/raw/vllm-0.10.1/`)
  is stripped first: it carries no signal and would only make every chunk
  longer. Together with the identifier subwords, this is the largest
  contributor to the score (see the ablation below).

## Performance analysis

Recall@5 from the official moulinette (`evaluate_student_search_results`,
`--k 10`): exact file-path match, span IoU > 0.05, matched references divided
by all references, averaged over every question. Before scoring it
validates the file and rejects any source wider than 2000 characters
(`--max_context_length`, default 2000). The project's own `evaluate`
command computes the same metric and gives identical numbers.

Scores on the public datasets:

| Dataset | Recall@5 | Threshold |
|---|---|---|
| docs | **0.880** | >= 0.80 |
| code | **0.869** | >= 0.50 |

Full recall curves (R@1 / R@3 / R@5 / R@10):

| Dataset | R@1 | R@3 | R@5 | R@10 |
|---|---|---|---|---|
| docs public | 0.67 | 0.84 | 0.88 | 0.94 |
| code public | 0.61 | 0.79 | 0.87 | 0.90 |

The code margin is wide (0.87 vs a 0.50 threshold). The docs margin is
narrower, 0.88 vs 0.80, and a 100-question set moves one point per
question. That is why every retrieval decision below was measured rather
than assumed.

**Speed** (4-core x86-64 CPU, 8 GB RAM):

| Step | Measured | Limit |
|---|---|---|
| `index` (1847 files with content -> 14,724 chunks) | 7–13 s | 300 s |
| `search_dataset` on 200 questions (two runs, incl. start-up) | 7–8 s | 90 s |
| `answer --k 10`, one question | 12–14 s | — |
| `answer_dataset`, 5 questions (batches of 4 + 1) | 70 s | — |

Answer timings assume a warm page cache; the first run after a reboot,
which reads the model weights from disk, takes 2–3 times as long.

Qwen3-0.6B loads in its checkpoint's bfloat16 (transformers' default
`dtype="auto"`), about 1.2 GB of weights. On a small machine where the model,
torch and the rest no longer fit in RAM, the process swaps and answering
slows down by more than an order of magnitude.

### Ablation: why each decision is there

Recall@5 on the public datasets with one decision changed at a time (same
metric as the moulinette):

| Config | docs public | code public |
|---|---|---|
| **shipped** | **88.0** | **86.9** |
| path indexed once instead of twice | 88.0 | 82.8 |
| no file path in the indexed text | 85.0 | 73.7 |
| no identifier subwords | 86.0 | 66.7 |
| base stopwords (no interrogatives) | 85.0 | 86.9 |
| no chunk overlap | 87.0 | 85.9 |
| no stemming | 86.0 | 81.8 |

The selection rule: with 100 questions per set, one question is one
point, so a change was kept only when it gained more than one question on
one set without losing on the other. Read against that rule:

- The file path and the identifier subwords are the clear wins, worth 13
  to 20 points on code. Writing the path twice instead of once adds
  another 4 points on code and leaves docs unchanged.
- The interrogative stopwords help docs by 3 points and leave code
  untouched.
- Stemming gains 2 points on docs and 5 on code: it aligns
  "scheduling"/"scheduler" and "cached"/"caching".
- Overlap gives 1 point on each set, which is within noise. It stays as
  cheap insurance against an identifier split across a chunk boundary,
  for an index 1.5% larger (14,724 chunks instead of 14,501).

### Chunk size

The subject allows chunks smaller than 2000 characters. Recall@5 for
smaller `--max_chunk_size` values, everything else shipped:

| `--max_chunk_size` | chunks | docs public | code public |
|---|---|---|---|
| **2000** (default) | 14,724 | **88.0** | **86.9** |
| 1500 | 19,782 | 87.0 | 86.9 |
| 1000 | 30,602 | 89.0 | 86.9 |
| 500 | 61,381 | 82.0 | 85.9 |

Down to 1000 characters the scores move by at most one question, which
is noise. At 500 they fall on both sets, the "too small" effect described
under Chunking strategy. 2000 stays: it is the subject's default, no
smaller size beats it by more than one question, and it gives the LLM the
most context per source.

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
  `main()`, which prints to stderr and returns exit code 1. Next to it,
  `main()` keeps one last-resort `except Exception` that prints the
  error type and message instead of a traceback, so an unforeseen
  library failure still exits cleanly with code 1. The
  alternative — returning `None` or error strings — was rejected because
  it forces every caller to re-check and lets errors flow onward as if
  they were data.
- **BM25 parameters left at library defaults** (k1=1.5, b=0.75). Tuning
  them was measured and gave nothing once chunk overlap was in place:
  k1=1.2, for example, scores 87.0 on public docs and 86.9 on public code,
  within one question of the defaults. So the defaults were kept rather
  than shipping an unjustifiable magic number.
- **Answer generation kept honest and reproducible.** Qwen3-0.6B via
  transformers, chat template applied with `enable_thinking=False`
  (otherwise the model spends its whole token budget inside a `<think>`
  block), greedy decoding (`do_sample=False`) for reproducible answers,
  `max_new_tokens=192`, left-padded batches of 4, and the prompt sliced
  off the generated ids before decoding.
- **Context capped at 10,000 characters, filled by retrieval rank.** The
  full top-k still goes into the JSON output; only what reaches the model
  is bounded, because on CPU the prefill cost scales with prompt length.
  10,000 is five full 2000-character chunks, so the top 5, the sources
  Recall@5 is judged on, always reach the model whole. An earlier
  6000-character cap was faster but let only the top 3 or 4 chunks
  through, and on 8% of questions the correctly retrieved source never
  reached the model.
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
- **CPU prefill speed.** On the development laptop, long contexts made a
  single answer take ~15 minutes, made worse by swapping. A context budget
  fixed it without changing what gets saved to disk: a single answer now
  takes 12–14 seconds on a 4-core, 8 GB machine.
- **Small datasets, noisy numbers.** With 100 questions per set, every
  point of recall is a single question, so it is easy to read noise as
  progress. The fix was methodological: change one
  thing at a time, measure both sets, and keep a change only when it gains
  more than one question without losing on the other set (see the
  ablation).
- **An "obviously right" idea that measured worse.** Capping chunks per file
  to diversify the top-k seemed like a free win, but scored 8–11 points
  *worse*: consecutive overlapping chunks of the right file cover different
  regions, and each one is an independent chance to clear the IoU bar.
  Diversification traded those chances away for hits in the wrong files.

## What I would improve

- **Semantic retrieval next to BM25.** Paraphrased questions ("how do I
  limit GPU memory") miss chunks that use other words
  (`gpu_memory_utilization`). Dense embeddings combined with BM25 would
  catch those.
- **A reranker on the top 50.** A small cross-encoder would reorder BM25's
  candidates by meaning. Some docs misses sit just outside the top 5.
- **Structure-aware boundaries for Python.** Split on whole functions and
  classes from the AST, so a chunk never starts mid-function.
- **Answer verification.** Check that each claim in the answer appears in
  the retrieved context, and fall back to "the context does not say" when
  it does not.

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
