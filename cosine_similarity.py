from sentence_transformers import util
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import faiss_ideal_points


def getEmbeddingsFromList(embedder, input_list):
	return [embedder.encode(d, convert_to_tensor=True, show_progress_bar=True) for d in input_list]


def getLexiconEmbeddings(embedder, lexicons):
	queries = [list(lexicon) for lexicon in lexicons]
	return getEmbeddingsFromList(embedder, queries)


def getNameEmbeddings(embedder, author_names):
	queries = list(author_names)
	return embedder.encode(queries, convert_to_tensor=True, show_progress_bar=True)


def computeCosineScore(description_embedding, author_embeddings, use_lexicon=False):
	# if comparing with all words from each lexicon
	if use_lexicon:
		cos_scores_all = []
		for author in author_embeddings:
			cos_scores = util.pytorch_cos_sim(description_embedding, author)[0]
			cos_scores_all.append(cos_scores.cpu())
		# get average score from each lexicon
		mean_scores = np.array([torch.mean(scores) for scores in cos_scores_all])
		return mean_scores
	# if only comparing with name of each author type
	else:
		cos_scores = util.pytorch_cos_sim(description_embedding, author_embeddings)[0]
		cos_scores = cos_scores.cpu().numpy()
		return cos_scores


def getSecondLabelFaiss(faiss_index, author_indices, author_info):
	label = [author_info.labels[i] for i, indices in enumerate(author_indices) if faiss_index in indices][0]
	return label


def getAverageScorePerAuthor(scores, author_indices, author_info):
	mean_scores = [np.mean(scores[list(indices)]) for indices in author_indices]
	return mean_scores



def cosineSim(tweets, embedder, author_info, use_lexicon=False):
	"""
	Labels the tweet's account description by author type by comparing the sentence embedding of the description with
	the sentence embedding of either
	a) only the name of each author type (i.e. 'doctor'), or
	b) averaged similarity of all words in each author type lexicon
	"""
	# description_embeddings = getEmbeddingsFromList(embedder, descriptions)
	# description_embeddings = np.load('embeddings/description_embeddings_2020-02.npy')
	description_embeddings = np.load('embeddings/description_embeddings_all.npy')
	print(description_embeddings.shape)
	print(len(tweets))

	assert description_embeddings.shape[0] == len(tweets)

	if use_lexicon:
		author_embeddings = getLexiconEmbeddings(embedder, author_info.lexicons)
	else:
		author_embeddings = getNameEmbeddings(embedder, author_info.names)

	# calculate cosine similarity between description and lexicon keywords
	# get max scores for histogram
	# scores_histogram = []
	tweets['label'] = np.nan
	tweets['label_score'] = np.nan
	thresholds = dict(zip(author_info.names, [0.45, 0.45, 0.45, 0.35]))
	print('Cosine similarities')
	for idx, description_embedding in enumerate(description_embeddings):
		scores = computeCosineScore(description_embedding, author_embeddings, use_lexicon)
		scores_dict = dict(zip(author_info.names, scores))
		label = max(scores_dict, key=scores_dict.get)
		# scores_histogram.append(scores_dict[label])
		if scores_dict[label] > thresholds[label]:
			tweets['label'].iloc[idx] = label
			tweets['label_score'].iloc[idx] = scores_dict[label]
			# print(scores_dict)
			# print('Original description: {} \nLabel: {} \nScore: {} \nIndex: {} \n'.format(tweets.iloc[idx].description,
																						# label, scores_dict[label], idx))
	print('Finished similarities')

	return tweets


def cosineSimSecondFaiss(tweets, author_info):
	"""
	"""
	# description_embeddings = getEmbeddingsFromList(embedder, descriptions)
	# description_embeddings = np.load('embeddings/description_embeddings_2020-02.npy')
	description_embeddings = np.load('embeddings/description_embeddings_all.npy')

	set_a = description_embeddings[np.where(tweets.label.notnull())[0]]
	set_b = description_embeddings[np.where(tweets.label.isnull())[0]]

	assert description_embeddings.shape[0] == len(set_a) + len(set_b)

	distances, indices = faiss_ideal_points.computeFaissSim(set_a, set_b)

	set_a_df = tweets[tweets.label.notnull()]
	set_b_df = tweets[tweets.label.isnull()]

	author_indices = [set(np.where(set_a_df.label == author)[0]) for author in author_info.names]

	set_b_df['second_cos_score'] = distances[:,0]
	set_b_df['second_cos_index'] = indices[:,0]
	set_b_df['second_cos_label'] = set_b_df.second_cos_index.apply(lambda x: getSecondLabel(x, author_indices, author_info))

	tweets = set_a_df.append(set_b_df.drop(['second_cos_index'], axis=1)).sort_index()
	tweets = tweets.loc[(tweets['second_cos_score'] < 160) | (tweets['label_score'].notnull())]

	return tweets


def cosineSimSecond(tweets, author_info):
	"""
	"""
	# description_embeddings = getEmbeddingsFromList(embedder, descriptions)
	# description_embeddings = np.load('embeddings/description_embeddings_2020-02.npy')
	description_embeddings = np.load('embeddings/description_embeddings_all.npy')

	set_a = description_embeddings[np.where(tweets.label.notnull())[0]]
	set_b = description_embeddings[np.where(tweets.label.isnull())[0]]

	assert description_embeddings.shape[0] == len(set_a) + len(set_b)

	set_a_df = tweets[tweets.label.notnull()]
	set_b_df = tweets[tweets.label.isnull()]

	author_indices = [set(np.where(set_a_df.label == author)[0]) for author in author_info.names]

	# calculate average cosine similarity between set_a (high quality) and set_b (lower quality)
	# apply second threshold to get set_c
	# get max scores for histogram
	# scores_histogram = []
	set_b_df['second_cos_label'] = np.nan
	set_b_df['second_cos_score'] = np.nan
	thresholds = dict(zip(author_info.names, [0.3, 0.35, 0.3, 0.25]))
	print(len(set_a))
	print(len(set_b))
	print('Cosine similarities')
	for idx, description_embedding in enumerate(tqdm(set_b)):
		scores = computeCosineScore(description_embedding, set_a)
		mean_scores = getAverageScorePerAuthor(scores, author_indices, author_info)
		scores_dict = dict(zip(author_info.names, mean_scores))
		label = max(scores_dict, key=scores_dict.get)
		# scores_histogram.append(scores_dict[label])
		if scores_dict[label] > thresholds[label]:
			set_b_df['second_cos_label'].iloc[idx] = label
			set_b_df['second_cos_score'].iloc[idx] = scores_dict[label]
	print('Finished similarities')

	print(set_b_df.second_cos_label.value_counts())
	for i, row in set_b_df.dropna(subset=['second_cos_label'])[:1000].iterrows():
		print(i)
		print(row.description)
		print(row.second_cos_label)
		print(row.second_cos_score, '\n')
	tweets = set_a_df.append(set_b_df).sort_index()
	print(len(tweets.dropna(subset=['second_cos_label'])))

	# clean
	tweets['label_score'].fillna(tweets['second_cos_score'], inplace=True)
	tweets['label'].fillna(tweets['second_cos_label'], inplace=True)
	tweets.dropna(subset=['label'], inplace=True)
	tweets.drop(labels=['second_cos_label', 'second_cos_score'], axis=1, inplace=True)

	return tweets

