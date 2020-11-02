import os
import numpy as np
import pandas as pd
import argparse
import utils
import cosine_similarity
import clustering
import re
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer, util



def readFiles(base_path, out_path=None):
	"""
	Reads and concatenates individual files into a dataframe
	"""
	ls = []
	for path, directory, file in os.walk(base_path):
	    for name in sorted(file):
	        if name.endswith('.csv') and '2020-02-06' in name:
	            filename = os.path.join(path, name)
	            df = pd.read_csv(filename, header=0, index_col=None, engine='python')
	            ls.append(df)
	tweets = pd.concat(ls, axis=0, ignore_index=True)
	if out_path:
		tweets.to_csv(out_path)
	return tweets


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--get_keyword_labels', default=False, help='Get descriptions corresponding to lexicon keyword matches')
	parser.add_argument('--model', default='clustering', help='Model to use to label account descriptions')

	args = parser.parse_args()

	base_path = 'data/tweets/'

	tweets = readFiles(base_path)
	print('# original tweets: {}'.format(len(tweets)))

	# drop NaNs
	tweets = tweets.dropna(subset=['description', 'tweet'])
	# remove Retweeted content
	tweets = tweets.drop(tweets[tweets.tweet.str.startswith('RT')].index)
	print('# tweets used (removed N/A + RTs): {}'.format(len(tweets)))

	# load author lexicons containing keywords
	academic_lexicon = set(open('tbip/lexicons/academic.txt', 'r').read().split())
	journalist_lexicon = set(open('tbip/lexicons/journalist.txt', 'r').read().split())
	doctor_lexicon = set(open('tbip/lexicons/doctor.txt', 'r').read().split())
	politician_lexicon = set(open('tbip/lexicons/politician.txt', 'r').read().split())

	author_lexicons = [academic_lexicon, journalist_lexicon, doctor_lexicon, politician_lexicon]
	author_names = ['academic', 'journalist', 'doctor', 'politician']
	author_labels = dict(enumerate(author_names))

	author_info = utils.authorInfo(author_names, author_labels, author_lexicons)

	# labeling account descriptions only by most lexicon keyword matches
	if args.get_keyword_labels:
		tweets_to_label = tweets.copy()
		tweets_with_keyword, n_labelled = utils.getKeywordLabels(tweets_to_label, author_info, equal_prob_flag=False, print_results=15)
		print('{:.0%} of account descriptions have keywords\n'.format(n_labelled/len(tweets)))

	stop_words = set(stopwords.words('english'))
	# tweets['description'] = tweets.description.apply(lambda x: utils.deEmojify(x))
	# tweets['description'] = tweets.description.apply(lambda x: utils.removeStopwords(x, stop_words))

	embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

	descriptions = list(tweets.description)

	# choose model
	if args.model == 'cosine_sim':
		cosine_similarity.cosineSim(descriptions, embedder, author_info, use_lexicon=False)
	if args.model == 'clustering':
		clustering.clustering(descriptions, embedder, len(author_names), author_lexicons, author_labels)


