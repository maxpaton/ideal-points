import os
import sys
import glob
import numpy as np
import pandas as pd
import argparse
import utils
import cosine_similarity
import clustering
import re
import matplotlib.pyplot as plt
import emoji
import bm25_similarity
import time



def readTweets(base_path, out_path=None):
	"""
	Reads and concatenates individual files into a dataframe
	"""
	ls = []
	for path, directory, file in os.walk(base_path):
	    for name in sorted(file):
	        # if name.endswith('.csv') and '2020-02' in name:
	        # if name.endswith('.csv') and '2020-01' in name:
	        # if name.endswith('.csv') and re.search('2020-0[12]', name):
	        if name.endswith('.csv'):
	            filename = os.path.join(path, name)
	            df = pd.read_csv(filename, header=0, index_col=None, engine='python')
	            ls.append(df)
	tweets = pd.concat(ls, axis=0, ignore_index=True)
	if out_path:
		tweets.to_csv(out_path)
	return tweets

def readLexicons(base_path):
	author_lexicons = []
	for file in sorted(glob.glob(base_path + '*.txt')):
		lexicon = set(open(file, 'r').read().split())
		author_lexicons.append(lexicon)
	return author_lexicons

def exportTweetsForBOW(tweets, out_path):
	# tweets.drop(['description', 'proba'], axis=1, inplace=True)
	tweets = tweets[['label', 'id', 'time', 'tweet']]
	tweets.columns = ['screen_name', 'id', 'created_at', 'text']
	tweets.to_csv(out_path, index=False)

def buildIndex(df):
	# convert descriptions to JSON
	df.columns = ['id', 'contents']
	df.to_json('docs_jsonl/documents.jsonl', orient='records', lines=True)
	# build index
	cmd = 'python3 -m pyserini.index -collection JsonCollection -generator DefaultLuceneDocumentGenerator \
				-threads 1 -input docs_jsonl \
				-index indexes/docs_jsonl -storePositions -storeDocvectors -storeRaw'
	os.system(cmd)


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--get_keyword_labels', default=False, help='Get descriptions corresponding to lexicon keyword matches')
	parser.add_argument('--model', required=True, help='Model to use to label account descriptions')

	args = parser.parse_args()

	base_path = 'data/tweets/'

	tweets = readTweets(base_path)
	print('# original tweets: {}'.format(len(tweets)))
	print(tweets.head())

	# drop NaNs
	tweets = tweets.dropna(subset=['description', 'tweet'])
	# drop retweeted content
	tweets = tweets.drop(tweets[tweets.tweet.str.startswith('RT')].index)
	print('# tweets used (removed N/A + RTs): {}'.format(len(tweets)))
	# convert tweets IDs to numeric
	tweets.id = pd.to_numeric(tweets.id)
	tweets = tweets.drop_duplicates(subset=['id'])
	# remove URLs
	tweets.tweet = tweets.tweet.apply(lambda x: re.sub(r'http\S+', '', x))


	# export descriptions to compute embeddings on Colab
	# tweets_temp = tweets.description
	# print(len(tweets_temp))
	# tweets_temp.to_csv('for_embeddings/tweets_for_embeddings.csv')
	# print('Tweets for embeddings saved')
	# sys.exit()

	# load author lexicons containing keywords
	author_lexicons = readLexicons('tbip/lexicons/')
	author_names = ['academic', 'doctor', 'journalist', 'politician']
	author_labels = dict(enumerate(author_names))

	author_info = utils.authorInfo(author_names, author_labels, author_lexicons)

	# labelling account descriptions only by most lexicon keyword matches
	if args.get_keyword_labels:
		tweets_to_label = tweets.copy()
		tweets_with_keyword, n_labelled = utils.getKeywordLabels(tweets_to_label, author_info, equal_prob_flag=False, print_results=20)
		print('{:.0%} of account descriptions have keywords\n'.format(n_labelled/len(tweets)))
		# exportTweetsForBOW(tweets_with_keyword, 'tbip/data/covid-tweets-2020/raw/tweets.csv')


	# descriptions = list(tweets.description.apply(lambda x: emoji.demojize(x, remove=True)))
	# choose model
	if args.model == 'cosine_sim':
		tweets = cosine_similarity.cosineSim(tweets, author_info, use_lexicon=False)
		tweets = cosine_similarity.cosineSimSecondFaiss(tweets, author_info)
		# tweets = cosine_similarity.cosineSimSecond(tweets, author_info)
		print(tweets)
		print(tweets.label.value_counts())
		# utils.plotSimilarityScores(scores_histogram)

	if args.model == 'clustering':
		# change so outputs tweets df
		scores = clustering.clusteringAll(tweets, embedder, author_info)

	if args.model == 'bm25':
		buildIndex(tweets[['id', 'description']])
		tweets = bm25_similarity.bm25Sim(tweets, author_info)
		tweets.dropna(inplace=True)
		print(len(tweets))
		print(tweets.label.value_counts())
	
	exportTweetsForBOW(tweets, 'tbip/data/covid-tweets-2020/raw/tweets_' + args.model + '.csv')




