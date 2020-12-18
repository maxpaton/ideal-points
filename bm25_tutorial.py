from bm25 import BM25


# we'll generate some fake texts to experiment with
corpus = [
    'Human machine interface for lab abc computer applications',
    'A survey of user opinion of computer system response time',
    'The EPS user interface management system',
    'System and human system engineering testing of EPS',
    'Relation of user perceived response time to error measurement',
    'The generation of random binary unordered trees',
    'The intersection graph of paths in trees',
    'Graph minors IV Widths of trees and well quasi ordering',
    'Graph minors A survey'
]

# remove stop words and tokenize them (we probably want to do some more
# preprocessing with our text in a real world setting, but we'll keep
# it simple here)
stopwords = set(['for', 'a', 'of', 'the', 'and', 'to', 'in'])
texts = [
    [word for word in document.lower().split() if word not in stopwords]
    for document in corpus
]

# build a word count dictionary so we can remove words that appear only once
word_count_dict = {}
for text in texts:
    for token in text:
        word_count = word_count_dict.get(token, 0) + 1
        word_count_dict[token] = word_count

texts = [[token for token in text if word_count_dict[token] > 1] for text in texts]
print(texts)

# query our corpus to see which document is more relevant
query = 'The intersection of graph survey and trees'
query = [word for word in query.lower().split() if word not in stopwords]

bm25 = BM25()
bm25.fit(texts)
scores = bm25.search(query)

for score, doc in zip(scores, corpus):
    score = round(score, 3)
    print(str(score) + '\t' + doc)