from pyserini.search import SimpleSearcher

# searcher = SimpleSearcher.from_prebuilt_index('robust04')
# hits = searcher.search('hubble space telescope')
searcher = SimpleSearcher('indexes/docs_jsonl')
hits = searcher.search('drugs')

# Print the first 10 hits:
for i in range(0, 10):
    print(f'{i+1:2} {hits[i].docid:15} {hits[i].score:.5f}')