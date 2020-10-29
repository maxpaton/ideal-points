import re
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
import numpy as np
import pandas as pd


def deEmojify(text):
	"""
	Removes emojis and other unicode standards from text
	"""
	regrex_pattern = re.compile(pattern = "["
	u"\U0001F600-\U0001F64F"  # emoticons
	u"\U0001F300-\U0001F5FF"  # symbols & pictographs
	u"\U0001F680-\U0001F6FF"  # transport & map symbols
	u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
						"]+", flags = re.UNICODE)
	return regrex_pattern.sub(r'', text)


def removeStopwords(text, stop_words):
	"""
	Removes stopwords, non-alphanumeric characters and lowercases text
	"""
	word_tokens = nltk.word_tokenize(text)
	removed = [w for w in word_tokens if w.lower() not in stop_words]
	removed = [re.sub(r'[^\s\w]', '', w) for w in removed]
	removed = [re.sub(r'\b\w\b', '', w) for w in removed]
	return ' '.join(removed)


def sentTokenize(text):
	nlp = spacy.load('en_core_web_sm')
	doc = nlp(text)
	return [sent.string.strip() for sent in doc.sents]


def lemmatize(text, lemmatizer):
	parsed = nltk.word_tokenize(text)
	return [lemmatizer.lemmatize(w) for w in parsed]


def lemmatizeLexicons(author_lexicons, lemmatizer):
	return [set([lemmatizer.lemmatize(w) for w in author]) for author in author_lexicons]


def getKWCounts(tweets, author_lexicons, col_names):
	for author, col_name in zip(author_lexicons, col_names):
		tweets[col_name + '_count'] = tweets.description.apply(lambda x: len([w for w in x if w in author]))
	return tweets


def getKWProba(tweets):
	counts = tweets.loc[:, 'academic_count': 'politician_count'].values
	proba = counts/counts.sum(axis=1).reshape(-1,1)
	tweets['proba'] = proba.tolist()
	return tweets


def getValidTweets(tweets, get_equal_prob):
	if get_equal_prob:
		mask = (tweets.proba.apply(lambda x: np.isfinite(x).all())) & (tweets.proba.apply(lambda x: x.count(max(x))>1))
	else:
		mask = (tweets.proba.apply(lambda x: np.isfinite(x).all())) & (tweets.proba.apply(lambda x: x.count(max(x))==1))
	return tweets[mask]


def getLabels(tweets, author_names):
	count_cols = tweets.loc[:, 'academic_count': 'politician_count']
	label = count_cols.idxmax(1)
	label_names = dict(zip(count_cols.columns, author_names))
	tweets['label'] = label.apply((lambda x: label_names[x]))
	return tweets


def printSelection(print_selection, tweets, tweets_orig, author_names, author_lexicons):
	# concat/compare original descriptions with proba and labels
	final = pd.concat([tweets_orig.loc[tweets.index].description, tweets[['proba', 'label']]], axis=1)
	for idx, row in final[:print_selection].iterrows():
		# get keywords present from each lexicon
		kws = [[w for w in tweets.loc[idx].description if w in author] for author in author_lexicons]
		print('Label: {} \nProba: {} \nDescription: {}'.format(row.label, row.proba, row.description))
		print(dict(zip(author_names, kws)), '\n')



def getKWLabels(tweets, author_lexicons, author_dict, get_equal_prob=False, print_selection=False):
	"""
	Returns account descriptions which contain an unambiguous maximum number of keywords from a single author lexicon

	param get_equal_prob: returns account descriptions which contain an equal number of keywords from multiple author lexicons, 
	and so are equally likely to correspond to multiple author types (i.e equal probability for multiple classes)
	prarm print_selection: prints print_selection descriptions to visually inspect each description and its corresponding lexicon keywords
	"""
	if not isinstance(print_selection, int) or print_selection == True:
		raise TypeError

	tweets_labelled = tweets.copy()

	# lemmatize descriptions and lexicons
	lemmatizer = WordNetLemmatizer()
	tweets_labelled['description'] = tweets_labelled.description.apply(lambda x: lemmatize(x.lower(), lemmatizer))
	author_lexicons_lemmatized = lemmatizeLexicons(author_lexicons, lemmatizer)

	# get keyword counts
	tweets_labelled = getKWCounts(tweets_labelled, author_lexicons_lemmatized, author_dict.values())

	tweets_labelled = getKWProba(tweets_labelled)

	# either descriptions with equal probability or unambiguous keyword matches
	tweets_labelled = getValidTweets(tweets_labelled, get_equal_prob)

	# get label corresponding to max count
	tweets_labelled = getLabels(tweets_labelled, author_dict.values())

	# print selection of records to visually inspect which keywords are present from each lexicon
	if print_selection:
		printSelection(print_selection, tweets_labelled, tweets, author_dict.values(), author_lexicons)

	return tweets_labelled[['description', 'proba', 'label']]




