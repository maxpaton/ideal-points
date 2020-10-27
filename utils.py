import re
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
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


def getTiedLabels(tweets, author_types, author_dict, lemmatize=True, print_selection: int=50):
	"""
	Returns account descriptions which contain an equal number of keywords (tied) from multiple author lexicons, 
	and so are equally likely to correspond to multiple author types (i.e equal probability for multiple classes)
	"""
	if not isinstance(print_selection, int):
		raise TypeError

	tweets_ = tweets.copy()

	lemmatizer = WordNetLemmatizer()
	tweets_['description'] = tweets_.description.apply(lambda x: nltk.word_tokenize(x))
	tweets_['description'] = tweets_.description.apply(lambda x: [w.lower() for w in x])
	tweets_['description'] = tweets_.description.apply(lambda x: [lemmatizer.lemmatize(w) for w in x])

	tweets_['academic_count'] = tweets_.description.apply(lambda x: len([w for w in x if w in author_types[0]]))
	tweets_['journalist_count'] = tweets_.description.apply(lambda x: len([w for w in x if w in author_types[1]]))
	tweets_['doctor_count'] = tweets_.description.apply(lambda x: len([w for w in x if w in author_types[2]]))
	tweets_['politician_count'] = tweets_.description.apply(lambda x: len([w for w in x if w in author_types[3]]))

	counts = tweets_.loc[:, 'academic_count': 'politician_count'].values
	proba = counts/counts.sum(axis=1).reshape(-1,1)
	tweets_['proba'] = proba.tolist()

	# choose label based on max author type count
	mask = (tweets_.proba.apply(lambda x: np.isfinite(x).all())) & (tweets_.proba.apply(lambda x: x.count(max(x))>1))
	valid = tweets_[mask]
	tweets_['label'] = 0
	tweets_.loc[mask, 'label'] = tweets_.loc[:, 'academic_count': 'politician_count'].idxmax(1)

	# encode categorical author type label
	d = {'academic_count': 'academic', 'journalist_count': 'journalist', 'doctor_count': 'doctor', 'politician_count': 'politician'}
	tweets_['label'] = tweets_.label.apply(lambda x: d[x] if x != 0 else x)
	# concat/compare original descriptions with proba and labels
	final = pd.concat([tweets[mask].description, tweets_[mask][['proba', 'label']]], axis=1)

	# print selection of records to visually inspect which keywords are present from each lexicon
	if print_selection:
		for idx, row in final[:print_selection].iterrows():
			# get keywords present from each lexicon
			kws = [[w for w in tweets_.loc[idx].description if w in author] for author in author_types]
			print(dict(zip(author_dict.values(), kws)))
			print('Label: {} \nProba: {} \nDescription: {} \n'.format(row.label, row.proba, row.description))

	return final


def getKWLabels(tweets, author_types, lemmatize=True):
	"""
	Returns account descriptions which contain an equal number of keywords (tied) from multiple author lexicons, 
	and so are equally likely to correspond to multiple author types
	"""
	tweets_ = tweets.copy()

	lemmatizer = WordNetLemmatizer()
	tweets_['description'] = tweets_.description.apply(lambda x: nltk.word_tokenize(x))
	tweets_['description'] = tweets_.description.apply(lambda x: [w.lower() for w in x])
	tweets_['description'] = tweets_.description.apply(lambda x: [lemmatizer.lemmatize(w) for w in x])

	tweets_['academic_count'] = tweets_.description.apply(lambda x: len([w for w in x if w in author_types[0]]))
	tweets_['journalist_count'] = tweets_.description.apply(lambda x: len([w for w in x if w in author_types[1]]))
	tweets_['doctor_count'] = tweets_.description.apply(lambda x: len([w for w in x if w in author_types[2]]))
	tweets_['politician_count'] = tweets_.description.apply(lambda x: len([w for w in x if w in author_types[3]]))

	counts = np.array([tweets_.academic_count, tweets_.journalist_count, tweets_.doctor_count, tweets_.politician_count]).T
	sums = tweets_.academic_count + tweets_.journalist_count + tweets_.doctor_count + tweets_.politician_count
	proba = counts/sums.values.reshape(-1,1)
	tweets_['proba'] = proba.tolist()

	# choose label based on max author type count
	mask = (pd.notnull(tweets_.proba.apply(lambda x: x[0]))) & (tweets_.proba.apply(lambda x: x.count(max(x))==1))
	valid = tweets_[mask]
	tweets_['label'] = 0
	tweets_.loc[mask, 'label'] = tweets_[['academic_count', 'journalist_count', 'doctor_count', 'politician_count']].idxmax(1)

	# encode categorical author type label
	d = {'academic_count': 'academic', 'journalist_count': 'journalist', 'doctor_count': 'doctor', 'politician_count': 'politician'}
	tweets_['label'] = tweets_.label.apply(lambda x: d[x] if x != 0 else x)
	tweets_final = tweets_.reset_index()

	return tweets_final.loc[tweets_final.label != 0][['description', 'proba', 'label']]






