import numpy as np
import pandas as pd
import os
import nltk
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer, util
import torch
import re
from nltk.corpus import stopwords
from itertools import chain
from sklearn.cluster import KMeans
import util


def cosineSim(descriptions, author_types, tweets_orig):
	
	print('Encoding descriptions')
	description_embeddings = [embedder.encode(d, convert_to_tensor=True) for d in descriptions]
	print('Finished encoding descriptions')
	queries = [list(author) for author in author_types]

	print('Encoding lexicons')
	kw_embeddings = [embedder.encode(author, convert_to_tensor=True) for author in queries]
	print('Finished encoding lexicons')

	print('Cosine similarities')
	author_pred = []
	for idx, description in enumerate(description_embeddings):
		cos_scores_all = []
		for author in kw_embeddings:
			cos_scores = util.pytorch_cos_sim(description, author)[0]
			cos_scores_all.append(cos_scores.cpu())
		mean_scores = [torch.mean(scores) for scores in cos_scores_all] 
		pred = max(mean_scores)
		print('Original description: {} \nPrediction: {} \nScore: {} \nIndex: {} \n'.format(list(tweets_orig.description)[idx], 
																				author_dict[mean_scores.index(max(mean_scores))], pred, idx))
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
		clustering_model = KMeans(n_clusters=num_clusters, n_init=5)
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
	print(len(tweets))

	# drop NaNs
	tweets = tweets.dropna(subset=['description', 'tweet'])
	# remove Retweeted content
	tweets = tweets.drop(tweets[tweets.tweet.str.startswith('RT')].index)
	print(len(tweets))

	tweets_orig = tweets.copy()

	lemmatizer = WordNetLemmatizer()

	academic_kw = set(open('tbip/lexicons/academic.txt', 'r').read().split())
	journalist_kw = set(open('tbip/lexicons/journalist.txt', 'r').read().split())
	doctor_kw = set(open('tbip/lexicons/doctor.txt', 'r').read().split())
	politician_kw = set(open('tbip/lexicons/politician.txt', 'r').read().split())

	# lemmatize (SHOULD PROBABLY ONLY DO IF LEMMATIZE DESCRIPTIONS TOO)
	academic_kw = set([lemmatizer.lemmatize(w) for w in academic_kw])
	journalist_kw = set([lemmatizer.lemmatize(w) for w in journalist_kw])
	doctor_kw = set([lemmatizer.lemmatize(w) for w in doctor_kw])
	politician_kw = set([lemmatizer.lemmatize(w) for w in politician_kw])

	author_types = [academic_kw, journalist_kw, doctor_kw, politician_kw]

	# tweets['description'] = tweets.description.apply(lambda x: nltk.word_tokenize(x))
	# tweets['description'] = tweets.description.apply(lambda x: [w.lower() for w in x])
	# tweets['description'] = tweets.description.apply(lambda x: [lemmatizer.lemmatize(w) for w in x])
	# tweets['description'] = tweets.description.apply(lambda x: ' '.join(x))

	stop_words = set(stopwords.words('english'))
	tweets['description'] = tweets.description.apply(lambda x: util.deEmojify(x))
	tweets['description'] = tweets.description.apply(lambda x: util.removeStopwords(x, stop_words))
	print(tweets)

	author_dict = {0: 'academic', 1: 'journalist', 2: 'doctor', 3: 'politician'}

	embedder = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')

	descriptions = list(tweets.description)

	# cosineSim(descriptions, author_types, tweets_orig)
	clustering(descriptions, author_types, author_dict, tweets_orig)


