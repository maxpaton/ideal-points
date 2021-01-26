from sentence_transformers import util
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyserini.search import SimpleSearcher



def descriptionsToJSON(df):
	df.columns = ['id', 'contents']
	df.to_json('docs_jsonl_cosine_sim/documents.jsonl', orient='records', lines=True)


def bm25Sim(tweets):#, author_info):
	"""
	"""
	# parameters
	fb_terms = 2
	fb_docs = 2
	original_query_weight = 0.9
	k_1 = 0.5
	b = 0.7
	k = 50000  # num hits

	searcher = SimpleSearcher('indexes/docs_jsonl_cosine_sim')
	# searcher.set_bm25(0.1, 0.9)
	searcher.set_rm3(fb_terms, fb_docs, original_query_weight)	

	# tweets['label'] = np.nan
	# tweets['label_score'] = np.nan

	# for author in author_info.names:
	hits = searcher.search('mask', k)
	print(len(hits))
	tweet_ids = [int(hit.docid) for hit in hits]
	scores = [hit.score for hit in hits]
	mask = tweets.id.isin(tweet_ids)
	# for i, row in tweets.loc[mask].iterrows():
	# 	print(row.text)

	return tweets[mask]


tweets = pd.read_csv('tbip/data/covid-tweets-2020/raw/tweets_cosine_sim.csv')

# convert tweets to JSON first
descriptionsToJSON(tweets[['id', 'text']])

######
# run following lunix command to build index:

# python3 -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
#  -threads 1 -input docs_jsonl_cosine_sim \
#  -index indexes/docs_jsonl_cosine_sim -storePositions -storeDocvectors -storeRaw

 # then uncomment below code
######

tweets = bm25Sim(tweets)
print(len(tweets))
tweets.dropna(inplace=True)
print(len(tweets))
# for i, row in tweets[:1000].iterrows():
# 	print('##############')
# 	print(row.text)
# 	print('##############')

print(tweets)

tweets.to_csv('tbip/data/covid-tweets-2020/raw/tweets_cosine_sim_masks.csv')