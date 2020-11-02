import numpy as np
from itertools import chain
from sklearn.cluster import KMeans



def clustering(descriptions, embedder, author_lexicons, author_labels, n_clusters):

	print('Encoding descriptions')
	description_embeddings = [embedder.encode(d) for d in descriptions]
	print('Finished encoding descriptions')

	# concat all lexicons
	all_lexicons = list(set(chain.from_iterable(author_lexicons)))
	lexicons_encoded = embedder.encode(all_lexicons)

	for idx, description in enumerate(description_embeddings):
		corpus = all_lexicons + [descriptions[idx]]
		corpus_embeddings = np.append(lexicons_encoded, description.reshape(1, -1), axis=0)

		# Perform kmean clustering
		num_clusters = 4
		clustering_model = KMeans(n_clusters=num_clusters, n_init=20)
		clustering_model.fit(corpus_embeddings)
		cluster_assignment = clustering_model.labels_

		clustered_sentences = [[] for i in range(num_clusters)]
		for sentence_id, cluster_id in enumerate(cluster_assignment):
			clustered_sentences[cluster_id].append(corpus[sentence_id])

		# get cluster which description falls into
		cluster_pred = [i for i, cluster in enumerate(clustered_sentences) if descriptions[idx] in cluster][0]
		# get counts of each author keywords in this cluster, then use author type with max. matches as author
		author_counts = [len(author & set(clustered_sentences[cluster_pred])) for author in author_lexicons]
		# print(author_counts)
		pred = author_labels[author_counts.index(max(author_counts))]
		# print(pred)
		print('Original description: {} \nPrediction: {} \nScore: {} \nIndex: {} \n'.format(descriptions[idx], 
																					author_counts, pred, idx))