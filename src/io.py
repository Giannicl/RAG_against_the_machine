import bm25s
import bm25s.high_level as bm25


corpus = bm25.load("./data/raw/benchmarks/sonnet.txt")

retriever = bm25.index(corpus)

queries = ["write a poetry about a treasure?", "what does a sonnet look like?"]
results = retriever.search(queries, k=5)

for result in results[0]:
    print(f"Score: {result['score']:.2f} | Document: {result['document']}")
