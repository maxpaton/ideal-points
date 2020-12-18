from sentence_transformers import util
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyserini.search import SimpleSearcher



def bm25Sim(tweets, author_info):
	"""
	"""
	# parameters
	fb_terms = 2
	fb_docs = 2
	original_query_weight = 1.0
	k = 50000

	searcher = SimpleSearcher('indexes/docs_jsonl')
	# searcher.set_bm25(0.3, 0.7)
	searcher.set_rm3(fb_terms, fb_docs, original_query_weight)	

	tweets['label'] = np.nan
	tweets['label_score'] = np.nan

	for author in author_info.names:
		hits = searcher.search(author, k)
		print(len(hits))
		tweet_ids = [int(hit.docid) for hit in hits]
		scores = [hit.score for hit in hits]
		mask = tweets.id.isin(tweet_ids)
		tweets.loc[mask, 'label'] = author_info.names[2]
		tweets.loc[mask, 'label_score'] = scores

	return tweets

	# tweets['label'] = np.nan
	# tweets['label_score'] = np.nan

	# searcher = SimpleSearcher('indexes/docs_jsonl')
	# # searcher.set_bm25(0.3, 0.7)
	# searcher.set_rm3(fb_terms, fb_docs, original_query_weight)

	# for author in author_info.names:
	# 	hits = searcher.search(author, k)
	# 	print(len(hits))
	# 	for i in range(len(hits)):
	# 		tweet_id = int(hits[i].docid)
	# 		score = hits[i].score
	# 		tweets['label'].loc[tweets['id'] == tweet_id] = author
	# 		tweets['label_score'].loc[tweets['id'] == tweet_id] = score

	# return tweets
