from sentence_transformers import util
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from BM25 import BM25
from nltk.corpus import stopwords



def BM25Sim(tweets, author_info):
	"""
	"""


	corpus = list(tweets.description)

	stop_words = set(stopwords.words('english'))

	texts = [
    [word for word in document.lower().split() if word not in stop_words]
    for document in corpus
	]

	# build a word count dictionary so we can remove words that appear only once
	word_count_dict = {}
	for text in texts:
	    for token in text:
	        word_count = word_count_dict.get(token, 0) + 1
	        word_count_dict[token] = word_count

	texts = [[token for token in text if word_count_dict[token] > 1] for text in texts]

	# query our corpus to see which document is more relevant
	query = 'politician'
	query = [word for word in query.lower().split() if word not in stop_words]

	bm25 = BM25()
	bm25.fit(texts)
	scores = bm25.search(query)

	for score, doc in zip(scores, corpus):
	    score = round(score, 3)
	    print(str(score) + '\t' + doc)






	# if use_lexicon:
	# 	author_embeddings = getLexiconEmbeddings(embedder, author_info.lexicons)
	# else:
	# 	author_embeddings = getNameEmbeddings(embedder, author_info.names)

	# # calculate cosine similarity between description and lexicon keywords
	# # get max scores for histogram
	# # scores_histogram = []
	# tweets['label'] = np.nan
	# tweets['label_score'] = np.nan
	# thresholds = dict(zip(author_info.names, [0.45, 0.45, 0.45, 0.35]))
	# print('Cosine similarities')
	# for idx, description_embedding in enumerate(description_embeddings):
	# 	scores = computeCosineScore(description_embedding, author_embeddings, use_lexicon)
	# 	scores_dict = dict(zip(author_info.names, scores))
	# 	label = max(scores_dict, key=scores_dict.get)
	# 	# scores_histogram.append(scores_dict[label])
	# 	if scores_dict[label] > thresholds[label]:
	# 		tweets['label'].iloc[idx] = label
	# 		tweets['label_score'].iloc[idx] = scores_dict[label]
	# 		# print(scores_dict)
	# 		# print('Original description: {} \nLabel: {} \nScore: {} \nIndex: {} \n'.format(tweets.iloc[idx].description,
	# 																					# label, scores_dict[label], idx))
	# print('Finished similarities')

	# return tweets