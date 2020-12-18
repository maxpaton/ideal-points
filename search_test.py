from pyserini.search import SimpleSearcher

fb_terms = 2
fb_docs = 2
original_query_weight = 1.0
k = 50000

searcher = SimpleSearcher('indexes/docs_jsonl')
searcher.set_rm3(fb_terms, fb_docs, original_query_weight)
hits = searcher.search('doctor', k)

# Print the first 10 hits:
for i in range(len(hits)):
    print(f'{i+1:2} {hits[i].docid:15} {hits[i].score:.5f}')
    print(searcher.doc(hits[i].docid).raw())