import spacy
import pandas as pd
import nltk
import re
from nltk.corpus import stopwords

data_path = 'tbip/data/covid-tweets-2020/raw/'
out_path = 'tbip/setup/stopwords/'

# def getMostFrequentWords(tweets, n_top):
# 	tweets_tokenized = [nltk.word_tokenize(tweet.lower()) for tweet in tweets]
#	# flatten
# 	tokens = [token for tweet in tweets_tokenized for token in tweet]
# 	# remove purely non-alphabetical tokens
# 	tokens = [re.sub('^[^A-Za-z]+$', '', token) for token in tokens]
# 	most_common = nltk.FreqDist(tokens).most_common(n_top)
# 	return [w for w,c in most_common]


def getMostFrequentWords(tweets, n_top):
	# tweets_tokenized = [nltk.word_tokenize(tweet.lower()) for tweet in tweets]
	tweets_tokenized = [tweet.lower().split(' ') for tweet in tweets]
	# print(tweets_tokenized)
	# flatten
	tokens = [token for tweet in tweets_tokenized for token in tweet]
	# remove purely non-alphabetical tokens
	tokens = [re.sub('^[^A-Za-z]+$', '', token) for token in tokens]
	most_common = nltk.FreqDist(tokens).most_common(n_top)
	custom_stopwords =  [w for w,c in most_common]
	# remove '' from stopwords (FIX THIS)
	custom_stopwords = [w for w in custom_stopwords if w]
	nltk_stopwords = stopwords.words('english')
	return set(custom_stopwords + nltk_stopwords)


if __name__ == '__main__':

	df = pd.read_csv(data_path + 'tweets.csv')
	tweets = df.text
	stopwords = getMostFrequentWords(list(tweets), 300)
	print(stopwords)
	with open(out_path + 'covid_tweets.txt', 'w') as file:
		file.write('\n'.join(stopwords))
