from sentence_transformers import util
import torch
import numpy as np
import pandas as pd
from collections import Counter
from tqdm import tqdm
from pyserini.search import SimpleSearcher


def assignScores(tweets, tweets_top, scores, author):
	'''
	Takes into account the fact that multiple docs could be assigned same author (it assigns the highest scoring author)
	'''
	for i, idx in enumerate(tweets_top.index):
		if pd.isnull(tweets.label_score.loc[idx]):
			tweets.label_score.loc[idx] = scores[i]
			tweets.label.loc[idx] = author
		elif scores[i] > tweets_top.label_score.loc[idx]:
			tweets.label_score.loc[idx] = scores[i]
			tweets.label.loc[idx] = author
		else:
			continue


def bm25Sim(tweets, author_info):
	"""
	No embeddings for this similarity method
	"""
	# parameters
	fb_terms = 2
	fb_docs = 2
	original_query_weight = 1.0
	k_1 = 0.5
	b = 0.7
	k = 50000	# num hits

	searcher = SimpleSearcher('indexes/docs_jsonl')
	# searcher.set_bm25(k_1, b)
	searcher.set_rm3(fb_terms, fb_docs, original_query_weight)	

	tweets['label'] = np.nan
	tweets['label_score'] = np.nan

	thresholds = dict(zip(author_info.names, [7, 8, 7, 8]))

	for i, author in enumerate(author_info.names):
		d = {}
		for word in author_info.lexicons[i]:
			hits = searcher.search(word, k)
			for hit in hits:
				if hit.docid in d:
					d[hit.docid] += hit.score
				else:
					d[hit.docid] = hit.score
		# get docs with scores higher than threshold
		# top_scores = list(filter(lambda x: x[1] > 6ld, d.items()))
		top_scores = list(filter(lambda x: x[1] > thresholds[author], d.items()))
		tweet_ids = [int(hit[0]) for hit in top_scores]
		print(len(tweet_ids))
		scores = [hit[1] for hit in top_scores]
		print(len(scores))
		mask = tweets.id.isin(tweet_ids)
		# tweets.loc[mask, 'label'] = author
		# tweets.loc[mask, 'label_score'] = scores
		tweets_top = tweets.loc[mask]
		assignScores(tweets, tweets_top, scores, author)

	for i, row in tweets.loc[tweets.label.notnull()][:1000].iterrows():
		print(row.description)
		print(row.label)
		print('\n')

	return tweets
