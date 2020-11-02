from sentence_transformers import util
import torch


def getEmbeddingsFromList(embedder, input_list):
	return [embedder.encode(d, convert_to_tensor=True) for d in input_list]


def getLexiconEmbeddings(embedder, lexicons):
	queries = [list(lexicon) for lexicon in lexicons]
	return getEmbeddingsFromList(embedder, queries)


def getNameEmbeddings(embedder, author_names):
	queries = list(author_names)
	return embedder.encode(queries, convert_to_tensor=True)


def getCosineScore(description_embedding, author_embeddings, use_lexicon):
	# if comparing with all words from each lexicon
	if use_lexicon:
		cos_scores_all = []
		for author in author_embeddings:
			cos_scores = util.pytorch_cos_sim(description_embedding, author)[0]
			cos_scores_all.append(cos_scores.cpu())
		# get average score from each lexicon
		mean_scores = [torch.mean(scores) for scores in cos_scores_all] 
		score = max(mean_scores)
		argmax = mean_scores.index(max(mean_scores))
		return score, argmax
	# if only comparing with name of each author type
	else:
		cos_scores = util.pytorch_cos_sim(description_embedding, author_embeddings)[0]
		cos_scores = cos_scores.cpu()
		score = torch.max(cos_scores)
		argmax = torch.argmax(cos_scores).item()
		return score, argmax


def cosineSim(descriptions, embedder, author_info, use_lexicon=False):
	"""
	Labels the tweet's account description by author type by comparing the sentence embedding of the description with
	the sentence embedding of either
	a) only the name of each author type (i.e. 'doctor'), or
	b) averaged similarity of all words in each author type lexicon
	"""
	print('Encoding descriptions')
	description_embeddings = getEmbeddingsFromList(embedder, descriptions)
	print('Finished encoding descriptions')

	print('Encoding keywords')
	if use_lexicon:
		author_embeddings = getLexiconEmbeddings(embedder, author_info.lexicons)
	else:
		author_embeddings = getNameEmbeddings(embedder, author_info.names)
	print('Finished keyword encoding')

	# calculate cosine similarity between description and lexicon keywords
	print('Cosine similarities')
	for idx, description_embedding in enumerate(description_embeddings):
		score, argmax = getCosineScore(description_embedding, author_embeddings, use_lexicon)
		label = author_info.labels[argmax]
		print('Original description: {} \nLabel: {} \nScore: {} \nIndex: {} \n'.format(descriptions[idx], 
																						label, score, idx))
	print('Finished encoding')

