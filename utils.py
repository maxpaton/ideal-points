import re
import nltk
# from nltk.stem import WordNetLemmatizer
import spacy
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
import torch
import emoji
import time


def demojize(text):
	"""
	Removes emojis and other unicode standards from text
	"""
	return emoji.demojize(text, remove=True)


def removeStopwords(text, stop_words):
	"""
	Removes stopwords, non-alphanumeric characters and lowercases text
	"""
	word_tokens = nltk.word_tokenize(text)
	removed = [w for w in word_tokens if w.lower() not in stop_words]
	removed = [re.sub(r'[^\s\w]', '', w) for w in removed]
	removed = [re.sub(r'\b\w\b', '', w) for w in removed]
	return ' '.join(removed)


def sentTokenize(text, lang_model):
	doc = lang_model(text)
	return [sent.string.strip() for sent in doc.sents]


def lemmatize(text, lang_model):
	doc = lang_model(text)
	return [token.lemma_ for token in doc]


def lemmatizeLexicons(lexicons, lang_model):
	return [set([lang_model(w)[0].lemma_ for w in author]) for author in lexicons]


# def lemmatize(text, lemmatizer):
# 	parsed = nltk.word_tokenize(text)
# 	return [lemmatizer.lemmatize(w) for w in parsed]

# def lemmatizeLexicons(lexicons, lemmatizer):
# 	return [set([lemmatizer.lemmatize(w) for w in author]) for author in lexicons]


def getKeywordCounts(tweets, author_info):
	for author, col_name in zip(author_info.lexicons_lemmatized, author_info.names):
		tweets[col_name + '_count'] = tweets.description_lemmatized.apply(lambda x: len([w for w in x if w in author]))
	return tweets


def getKeywordProba(tweets):
	counts = tweets.loc[:, 'academic_count': 'politician_count'].values
	proba = counts/counts.sum(axis=1).reshape(-1,1)
	tweets['proba'] = proba.tolist()
	return tweets


def getValidTweets(tweets, equal_prob_flag):
	if equal_prob_flag:
		mask = (tweets.proba.apply(lambda x: np.isfinite(x).all())) & (tweets.proba.apply(lambda x: x.count(max(x))>1))
	else:
		mask = (tweets.proba.apply(lambda x: np.isfinite(x).all())) & (tweets.proba.apply(lambda x: x.count(max(x))==1))
	return tweets[mask]


def getLabelsFromCounts(tweets, author_names):
	count_cols = tweets.loc[:, 'academic_count': 'politician_count']
	label = count_cols.idxmax(1)
	label_names = dict(zip(count_cols.columns, author_names))
	tweets['label'] = label.apply((lambda x: label_names[x]))
	return tweets


def label(tweets, author_info, equal_prob_flag):
	# get keyword counts and probability vector
	tweets = getKeywordCounts(tweets, author_info)
	tweets = getKeywordProba(tweets)
	# either descriptions with equal probability or unambiguous keyword matches
	tweets = getValidTweets(tweets, equal_prob_flag)
	# get label corresponding to max count
	return getLabelsFromCounts(tweets, author_info.names)


def printResults(print_results, tweets, author_info):
	# compare original descriptions with proba and labels
	for idx, row in tweets[['description', 'description_lemmatized', 'proba', 'label']][:print_results].iterrows():
		# get keywords present from each lexicon
		keywords = [[w for w in row.description_lemmatized if w in author] for author in author_info.lexicons_lemmatized]
		print('Label: {} \nProba: {} \nDescription: {} \nIndex: {}'.format(row.label, row.proba, row.description, idx))
		print(dict(zip(author_info.names, keywords)), '\n')


def getKeywordLabels(tweets, author_info, equal_prob_flag=False, print_results=False):
	"""
	Returns account descriptions which contain an unambiguous maximum number of keywords from a single author lexicon

	param equal_prob_flag: returns account descriptions which contain an equal number of keywords from multiple author lexicons, 
	and so are equally likely to correspond to multiple author types (i.e equal probability for multiple classes)
	param print_results: prints print_results descriptions to visually inspect each description and its corresponding lexicon keywords
	"""
	if not isinstance(print_results, int) or print_results == True:
		raise TypeError

	# lemmatize descriptions and lexicons
	nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'parser'])

	print('Started lemmatizing')
	start = time.time()
	tweets['description_lemmatized'] = tweets.description.apply(lambda x: lemmatize(x.lower(), nlp))
	author_info.lexicons_lemmatized = lemmatizeLexicons(author_info.lexicons, nlp)
	print('Took {} seconds'.format(time.time() - start))
	print('Finished lemmatizing')
	tweets = label(tweets, author_info, equal_prob_flag)

	# print selection of records to visually inspect which keywords are present from each lexicon
	if print_results:
		printResults(print_results, tweets, author_info)

	print(tweets)

	return tweets[['id', 'tweet', 'time', 'description', 'proba', 'label']], len(tweets)



class authorInfo():

	def __init__(self, names, labels, lexicons):
		self.names = names
		self.labels = labels
		self.lexicons = lexicons



# class dataEmbedder():

# 	def __init__(self, embedder, author_lexicons, author_labels):
# 		self.embedder = embedder
# 		self.author_lexicons = author_lexicons
# 		self.author_labels = author_labels

# 	def embed_descriptions(self, descriptions):

# 		print('Embedding descriptions')
# 		description_embeddings = [self.embedder.encode(d, convert_to_tensor=True) for d in descriptions]
# 		return description_embeddings, 'Finished embedding descriptions'

# 	def embed_lexicons(self, use_lexicon=False):
# 		# use all words from lexicons
# 		if use_lexicon:
# 			queries = [list(author) for author in self.author_lexicons]
# 			return [self.embedder.encode(author, convert_to_tensor=True) for author in queries]
# 		# use only author type name
# 		else:
# 			queries = self.author_labels.values()
# 			return self.embedder.encode(queries, convert_to_tensor=True)



# class clusteringSimilarity():

# 	def __init__(encoder, num_clusters):
# 		self.encoder = encoder
# 		self.num_clusters = num_clusters


# 	def encode_descriptions():


# 	def encode_lexicons():


# 	def calculateSimilarity():


# 	def print_results():



