import re
import nltk
import spacy
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import time


class authorInfo():

	def __init__(self, names, labels, lexicons):
		self.names = names
		self.labels = labels
		self.lexicons = lexicons


def plotSimilarityScores(scores):
	fig, ax = plt.subplots()
	ax.hist(scores, bins=20, edgecolor='black')
	ax.set_xlabel('Cosine similarity scores')
	ax.set_ylabel('Frequency')
	ax.yaxis.grid()
	fig.savefig('plots/cosine_similarity_hist.png', bbox_inches='tight', dpi=400)


def removeStopwords(text, stop_words):
	"""
	Removes stopwords, non-alphanumeric characters and lowercases text
	"""
	word_tokens = nltk.word_tokenize(text)
	removed = [w for w in word_tokens if w.lower() not in stop_words]
	removed = [re.sub(r'[^\s\w]', '', w) for w in removed]
	removed = [re.sub(r'\b\w\b', '', w) for w in removed]
	return ' '.join(removed)


# def sentTokenize(text, lang_model):
# 	doc = lang_model(text)
# 	return [sent.string.strip() for sent in doc.sents]


def lemmatize(text, nlp):
	doc = nlp(text)
	return [token.lemma_ for token in doc]


def getValidTweets(tweets, equal_prob_flag):
	if equal_prob_flag:
		mask = (tweets.proba.apply(lambda x: np.isfinite(x).all())) & (tweets.proba.apply(lambda x: x.count(max(x))>1))
	else:
		mask = (tweets.proba.apply(lambda x: np.isfinite(x).all())) & (tweets.proba.apply(lambda x: x.count(max(x))==1))
	return tweets[mask]


def label(tweets, author_info, equal_prob_flag):
	# get keyword counts
	for author, col_name in zip(author_info.lexicons_lemmatized, author_info.names):
		tweets[col_name + '_count'] = tweets.description_lemmatized.apply(lambda x: len([w for w in x if w in author]))

	# get keyword probability vector based on counts
	counts = tweets.loc[:, 'academic_count': 'politician_count'].values
	proba = counts/counts.sum(axis=1).reshape(-1,1)
	tweets['proba'] = proba.tolist()

	# either descriptions with equal probability or unambiguous keyword matches
	tweets = getValidTweets(tweets, equal_prob_flag)

	# get label corresponding to max count
	count_cols = tweets.loc[:, 'academic_count': 'politician_count']
	label = count_cols.idxmax(1)
	label_names = dict(zip(count_cols.columns, author_info.names))
	tweets['label'] = label.apply((lambda x: label_names[x]))

	return tweets


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
	+ equal_prob_flag: returns account descriptions which contain an equal number of keywords from multiple author lexicons, 
		and so are equally likely to correspond to multiple author types (i.e equal probability for multiple classes)
	+ print_results: prints print_results descriptions to visually inspect each description and its corresponding lexicon keywords
	"""
	if not isinstance(print_results, int) or print_results == True:
		raise TypeError

	# lemmatize descriptions and lexicons
	nlp = spacy.load('en_core_web_sm', disable=['tagger', 'ner', 'parser'])

	print('Started lemmatizing')
	tweets['description_lemmatized'] = tweets.description.apply(lambda x: lemmatize(x.lower(), nlp))
	author_info.lexicons_lemmatized = [set([nlp(w)[0].lemma_ for w in author]) for author in author_info.lexicons]
	print('Finished lemmatizing')
	tweets = label(tweets, author_info, equal_prob_flag)

	# print selection of records to visually inspect which keywords are present from each lexicon
	if print_results:
		printResults(print_results, tweets, author_info)

	print(tweets)
	counts = tweets.label.value_counts()
	print(counts/counts.sum())

	return tweets[['id', 'tweet', 'time', 'description', 'proba', 'label']], len(tweets)

