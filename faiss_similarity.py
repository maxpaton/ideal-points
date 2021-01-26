import faiss
import numpy as np


def computeFaissSim(db_vectors, query_vectors):
	# embedding dimension
	dim = db_vectors.shape[1]
	# assigns the vectors to a particular cluster
	quantiser = faiss.IndexFlatL2(dim)
	# specify number of clusters to be formed
	nlist = 4
	# here we specify METRIC_L2, by default it performs inner-product search
	index = faiss.IndexIVFFlat(quantiser, dim, nlist, faiss.METRIC_L2)

	# train on the database vectors
	index.train(db_vectors)
	# add the vectors and update the index
	index.add(db_vectors)
	print(f'index.is_trained: {index.is_trained}')
	print(f'vectors added to index: {index.ntotal}')

	# perform search on index for query embeddings

	# number of most similar clusters to find (must be < nlist)
	nprobe = 4
	# number of query embeddings
	n_query = query_vectors.shape[0]
	# number of nearest neighbours to return
	k = 1

	# search
	# returns the ids (row numbers or index in the vector store) 
	# of the k most similar vectors for each query vector along with their respective distances
	distances, indices = index.search(query_vectors, k)
	# print(f'distances: \n{distances}')
	# print(f'indices: \n{indices}')
	print("Finished Faiss search")

	return distances, indices




