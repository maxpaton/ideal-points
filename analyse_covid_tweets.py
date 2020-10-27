import numpy as np
import pandas as pd
import os
import argparse
import nltk
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import torch
import re
from nltk.corpus import stopwords
from itertools import chain
from sklearn.cluster import KMeans
import utils


def cosineSim(descriptions, author_types, authors, tweets_orig, use_entire_lexicon=False):
	"""
	Labels the tweet's account description by author type by comparing the sentence embedding of the description with
	the sentence embedding of either
	a) only the name of each author type (i.e. 'doctor'), or
	b) averaged similarity of all words in each author type lexicon
	"""
	print('Encoding descriptions')
	description_embeddings = [embedder.encode(d, convert_to_tensor=True) for d in descriptions]
	print('Finished encoding descriptions')

	print('Encoding lexicons')
	# use all words from lexicons
	if use_entire_lexicon:
		queries = [list(author) for author in author_types]
		kw_embeddings = [embedder.encode(author, convert_to_tensor=True) for author in queries]
	# use only author type name
	else:
		queries = authors
		kw_embeddings = embedder.encode(queries, convert_to_tensor=True)
	print('Finished encoding lexicons')

	# calculate cosine similarity between description and lexicon keywords
	print('Cosine similarities')
	for idx, description in enumerate(description_embeddings):
		# if comparing with all words from each lexicon
		if use_entire_lexicon:
			cos_scores_all = []
			for author in kw_embeddings:
				cos_scores = util.pytorch_cos_sim(description, author)[0]
				cos_scores_all.append(cos_scores.cpu())
			mean_scores = [torch.mean(scores) for scores in cos_scores_all] 
			pred = max(mean_scores)
			argmax = mean_scores.index(max(mean_scores))
		# if only comparing with name of each author type
		else:
			cos_scores = util.pytorch_cos_sim(description, kw_embeddings)[0]
			cos_scores = cos_scores.cpu()
			pred = torch.max(cos_scores)
			argmax = torch.argmax(cos_scores).item()
		print('Original description: {} \nPrediction: {} \nScore: {} \nIndex: {} \n'.format(list(tweets_orig.description)[idx], 
																						author_dict[argmax], pred, idx))
	print('Finished encoding')


def clustering(descriptions, author_types, author_dict, tweets_orig):

	print('Encoding descriptions')
	description_embeddings = [embedder.encode(d) for d in descriptions]
	print('Finished encoding descriptions')

	all_lexicons = list(set(chain.from_iterable(author_types)))
	lexicons_encoded = embedder.encode(all_lexicons)

	for idx, description in enumerate(description_embeddings):
		corpus = all_lexicons + [descriptions[idx]]
		corpus_embeddings = np.append(lexicons_encoded, description.reshape(1, -1), axis=0)

		# Perform kmean clustering
		num_clusters = 4
		clustering_model = KMeans(n_clusters=num_clusters, n_init=20)
		clustering_model.fit(corpus_embeddings)
		cluster_assignment = clustering_model.labels_

		clustered_sentences = [[] for i in range(num_clusters)]
		for sentence_id, cluster_id in enumerate(cluster_assignment):
			clustered_sentences[cluster_id].append(corpus[sentence_id])

		# get cluster which description falls into
		cluster_pred = [i for i, cluster in enumerate(clustered_sentences) if descriptions[idx] in cluster][0]
		# get counts of each author keywords in this cluster, then use author type with max. matches as author
		author_counts = [len(author & set(clustered_sentences[cluster_pred])) for author in author_types]
		# print(author_counts)
		pred = author_dict[author_counts.index(max(author_counts))]
		# print(pred)
		print('Original description: {} \nPrediction: {} \nScore: {} \nIndex: {} \n'.format(list(tweets_orig.description)[idx], 
																					author_counts, pred, idx))


if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	parser.add_argument('--get_equal_prob', default=False, help='Inspect the account descriptions with equal probability for multiple classes')
	parser.add_argument('--model', default='clustering', help='Which model to use to label account descriptions')

	args = parser.parse_args()

	out_path = 'tbip/data/covid-tweets-2020/raw/covid_tweets.csv'
	base_path = 'data/tweets/'

	ls = []
	for path, directory, file in os.walk(base_path):
	    for name in sorted(file):
	        if name.endswith('.csv') and '2020-02-06' in name:
	            filename = os.path.join(path, name)
	            df = pd.read_csv(filename, header=0, index_col=None, engine='python')
	            ls.append(df)

	tweets = pd.concat(ls, axis=0, ignore_index=True)
	print('# original tweets: {}'.format(len(tweets)))

	# drop NaNs
	tweets = tweets.dropna(subset=['description', 'tweet'])
	# remove Retweeted content
	tweets = tweets.drop(tweets[tweets.tweet.str.startswith('RT')].index)
	print('# tweets (removed N/A + RTs): {}'.format(len(tweets)))

	tweets_orig = tweets.copy()

	academic_kw = set(open('tbip/lexicons/academic.txt', 'r').read().split())
	journalist_kw = set(open('tbip/lexicons/journalist.txt', 'r').read().split())
	doctor_kw = set(open('tbip/lexicons/doctor.txt', 'r').read().split())
	politician_kw = set(open('tbip/lexicons/politician.txt', 'r').read().split())

	# lemmatize (SHOULD PROBABLY ONLY DO IF LEMMATIZE DESCRIPTIONS TOO)
	lemmatizer = WordNetLemmatizer()
	academic_kw = set([lemmatizer.lemmatize(w) for w in academic_kw])
	journalist_kw = set([lemmatizer.lemmatize(w) for w in journalist_kw])
	doctor_kw = set([lemmatizer.lemmatize(w) for w in doctor_kw])
	politician_kw = set([lemmatizer.lemmatize(w) for w in politician_kw])

	author_types = [academic_kw, journalist_kw, doctor_kw, politician_kw]
	authors = ['academic', 'journalist', 'doctor', 'politician']
	author_dict = dict(enumerate(authors))

	if args.get_equal_prob:
		print(utils.getTiedLabels(tweets, author_types, author_dict, print_selection=10))

	# tweets['description'] = tweets.description.apply(lambda x: nltk.word_tokenize(x))
	# tweets['description'] = tweets.description.apply(lambda x: [w.lower() for w in x])
	# tweets['description'] = tweets.description.apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
	# tweets['description'] = tweets.description.apply(lambda x: ' '.join(x))

	stop_words = set(stopwords.words('english'))
	tweets['description'] = tweets.description.apply(lambda x: utils.deEmojify(x))
	tweets['description'] = tweets.description.apply(lambda x: utils.removeStopwords(x, stop_words))
	print(tweets)

	embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

	descriptions = list(tweets.description)

	# choose model
	if args.model == 'cosine_sim':
		cosineSim(descriptions, author_types, authors, tweets_orig, use_entire_lexicon=True)
	if args.model == 'clustering':
		clustering(descriptions, author_types, author_dict, tweets_orig)


