import numpy as np
from copy import copy
from pytorch_sparse_regressor import PytorchSparseRegressor
import torch


def fit_all_vectors_pytorch(vectors, basis, alpha, reconstructed=False):
    # Create an index of indices to disallow because they map a basis vector to itself
    excluded_index = np.asarray([[vectors.vocab[word].index, i] for i, word in enumerate(basis.get_words_list())])

    sparse_vectors = copy(vectors)

    # Break n_vectors into approximately equal-sized chunks to address memory constraints
    n_vectors = vectors.syn0.shape[0]
    max_chunk_size = 12500
    chunk_sizes = [max_chunk_size] * (n_vectors // max_chunk_size)
    if sum(chunk_sizes) != n_vectors:
        chunk_sizes.append(n_vectors - sum(chunk_sizes))

    # Now irterate through all these chunks
    i = 0
    r = []
    for chunk in chunk_sizes:
        model = PytorchSparseRegressor(alpha, len(basis.get_words_list()), chunk)

        # Filter list of excluded items and change indexing
        excluded_filtered = excluded_index
        excluded_filtered[:, 0] = excluded_index[:, 0] - i
        excluded_filtered = excluded_filtered[
                            np.logical_and(excluded_filtered[:, 0] >= 0, excluded_filtered[:, 0] < chunk), :]

        r.append(model.fit(basis.get_matrix(), vectors.syn0[i:i + chunk, :], .002, 15000, excluded=excluded_filtered,
                           return_reconstructed=reconstructed))

        # manual garbage collection or old GPU memory will stick around (and nottrigger GC because RAM is still free)
        del model
        gc.collect()

        i += chunk

    sparse_vectors.syn0 = np.concatenate(r, axis=0)
    return sparse_vectors
