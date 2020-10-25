import re
import nltk



def deEmojify(text):
	regrex_pattern = re.compile(pattern = "["
	u"\U0001F600-\U0001F64F"  # emoticons
	u"\U0001F300-\U0001F5FF"  # symbols & pictographs
	u"\U0001F680-\U0001F6FF"  # transport & map symbols
	u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
						"]+", flags = re.UNICODE)
	return regrex_pattern.sub(r'', text)
	

def removeStopwords(text, stop_words):
	word_tokens = nltk.word_tokenize(text)
	removed = [w for w in word_tokens if w.lower() not in stop_words]
	removed = [re.sub(r'[^\s\w]', '', w) for w in removed]
	removed = [re.sub(r'\b\w\b', '', w) for w in removed]
	return ' '.join(removed)