import numpy as np
from itertools import chain
from sklearn.cluster import KMeans
import utils


def getEmbeddingsFromList(embedder, input_list):
	return [embedder.encode(d, show_progress_bar=True) for d in input_list]


def clusteringOld(tweets, embedder, author_info, n_clusters=4):

	
	print('Encoding descriptions')
	description_embeddings = getEmbeddingsFromList(descriptions)
	print('Finished encoding descriptions')

	# concat all lexicons
	all_lexicons = list(set(chain.from_iterable(author_info.lexicons)))
	lexicons_encoded = embedder.encode(all_lexicons)

	for idx, description in enumerate(description_embeddings):
		corpus = all_lexicons + [descriptions[idx]]
		# cluster the description + all individual lexicon words
		corpus_embeddings = np.append(lexicons_encoded, description.reshape(1, -1), axis=0)

		# Perform kmean clustering
		clustering_model = KMeans(n_clusters=n_clusters, n_init=10)
		clustering_model.fit(corpus_embeddings)
		cluster_assignment = clustering_model.labels_

		clustered_sentences = [[] for i in range(n_clusters)]
		for sentence_id, cluster_id in enumerate(cluster_assignment):
			clustered_sentences[cluster_id].append(corpus[sentence_id])

		# get cluster which description falls into
		cluster_pred = [i for i, cluster in enumerate(clustered_sentences) if descriptions[idx] in cluster][0]
		# get counts of each author keywords in this cluster, then use author type with max. matches as author
		author_counts = [len(author & set(clustered_sentences[cluster_pred])) for author in author_info.lexicons]
		label = author_info.labels[author_counts.index(max(author_counts))]
		print('Original description: {} \nPrediction: {} \nScore: {} \nIndex: {} \n'.format(descriptions[idx], 
																					author_counts, label, idx))

def clusteringAll(tweets, embedder, author_info, n_clusters=4):

	# descriptions = list(tweets.description)
	# description_embeddings = getEmbeddingsFromList(descriptions)
	# corpus_embeddings = embedder.encode(descriptions, show_progress_bar=True)
	# np.save('description_embeddings_2020-02_wo_emojis.npy', corpus_embeddings)

	description_embeddings = np.load('embeddings/description_embeddings_2020-02.npy')

	all_lexicons = list(set(chain.from_iterable(author_info.lexicons)))
	lexicon_embeddings = embedder.encode(all_lexicons)
	embeddings = np.append(description_embeddings, lexicon_embeddings, axis=0)

	descriptions_and_lexicons = descriptions + all_lexicons

	# Perform kmean clustering
	clustering_model = KMeans(n_clusters=n_clusters, n_init=20)
	clustering_model.fit(embeddings)
	cluster_assignment = clustering_model.labels_

	clustered_sentences = [[] for i in range(n_clusters)]
	for sentence_id, cluster_id in enumerate(cluster_assignment):
		clustered_sentences[cluster_id].append(descriptions_and_lexicons[sentence_id])

	for i, cluster in enumerate(clustered_sentences):
		print("Cluster {} ({})".format(i+1, len(cluster)))
		# print(cluster)
		print("")

	# squared distance from closest cluster centre
	X_dist = clustering_model.transform(embeddings)**2
	dist_to_cluster_center = np.choose(cluster_assignment, X_dist.T)
	for i, dist in enumerate(dist_to_cluster_center):
		print(descriptions_and_lexicons[i])
		print(dist, cluster_assignment[i], '\n')


def clustering(tweets, embedder, author_info, n_clusters=4):

	descriptions = list(tweets.description)
	print('Encoding descriptions')
	# description_embeddings = getEmbeddingsFromList(descriptions)
	corpus_embeddings = embedder.encode(descriptions)
	print('Finished encoding descriptions')

	all_lexicons = list(set(chain.from_iterable(author_info.lexicons)))
	lexicons_encoded = embedder.encode(all_lexicons)

	# Perform kmean clustering
	clustering_model = KMeans(n_clusters=n_clusters, n_init=50)
	clustering_model.fit(lexicons_encoded)
	cluster_assignment = clustering_model.labels_

	clustered_sentences = [[] for i in range(n_clusters)]
	for sentence_id, cluster_id in enumerate(cluster_assignment):
		clustered_sentences[cluster_id].append(all_lexicons[sentence_id])

	for i, cluster in enumerate(clustered_sentences):
		print("Cluster ", i+1)
		print(cluster)
		print("")

	for idx, description in enumerate(corpus_embeddings):
		print(descriptions[idx])
		print(clustering_model.predict(description.reshape(1,-1)))
		print(clustering_model.score(description.reshape(1,-1)))


	# print(clustering_model.score(corpus_embeddings[0].reshape(1,-1)))
	# print(clustering_model.predict(corpus_embeddings[0].reshape(1,-1)))
