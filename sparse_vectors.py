import gensim
import numpy as np
import gc
from gensim.models import KeyedVectors
from copy import copy
import multiprocessing
import sklearn
import functools
import lightning
from lightning.regression import FistaRegressor
import spacy
from sklearn.preprocessing import normalize
import math
import scipy
import tqdm

# Load pretrained fastText vectors

spacy_nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])


class Basis:
    def __init__(self, matrix, words_list=None, gensim_vocab=None):
        """We can construct from either a list of words or a gensim Vocab object, but not both
        matrix is a ndarray"""
        assert isinstance(matrix, np.ndarray)
        self.matrix = matrix
        assert (words_list is None) != (gensim_vocab is None)
        if words_list is not None:
            self.words_list = words_list
            self.words_inv = {word: i for i, word in enumerate(words_list)}
        elif gemsim_vocab is not None:
            self.words_inv = {word: gensim_vocab['word'].index for word in gensim_vocab.keys()}
            self.words_list = [self.words_inv[i] for i in range(len(self.words_inv))]
        assert self.matrix.shape[0] == len(self.words_list)

    def get_words_list(self):
        return self.words_list

    def get_words_inverse_map(self):
        return self.words_inv

    def get_matrix(self):
        return self.matrix

    def get_vector(self, word):
        return self.matrix[self.words_inv[word]]


def filter_by_lemma(vocab_dict, exclude=set()):
    """Filters a gensim vocabulary dict to contain only one word for each lemma
    Keeps the most frequent word
    Uses spacy lemmatizer
    Args:
        vocab: A vocabulary dictionary, such as one in a KeyedVector.vocab
    """
    # Maps lemmas to (word, vocabulary object) tuples
    # ie. the key-value tuples from vocab_dict
    # Weird one liner to get spacy lemma of a single word

    lemmatize = lambda word: spacy_nlp(word)[0].lemma_
    excluded_lemmas = {lemmatize(w) for w in exclude}
    lemma_dict = {}
    for word, vocab in vocab_dict.items():
        lemma = lemmatize(word)
        # We also remove capitalized forms
        if word.islower():
            if lemma not in lemma_dict or lemma_dict[lemma][1].count < vocab.count:
                lemma_dict[lemma] = (word, vocab)
    # Rewrap the values of lemma_dict as a dictionary to match original formatting
    return {k: v for k, v in lemma_dict.values()}


def get_top_n_vectors(vectors, n, exclude, do_filter_by_lemma=True, do_normalize=True):
    """Takes a gensim KeyedVectors model and an integer, n
    Args:
        vectors (KeyedModel): The model to extract
        n (int): The number of vectors to retrieve

    Returns:
        list[string]: The top n words, in the same order as the matrix
        ndarray: A numpy matrix containing all the vectors, indexed [word][dimension]
    """

    # Note that this assumes keys are in order for performance
    # True for gensim vectors, which are not actual python dicts
    vocab_filtered = {k: v for k, v in vectors.vocab.items() if v.index < n}
    if do_filter_by_lemma:
        vocab_filtered = filter_by_lemma(vocab_filtered, exclude=exclude)
    # Remove excluded words
    vocab_filtered = {k: v for k, v in vocab_filtered.items() if k not in exclude}
    words = list(vocab_filtered.keys())
    m = np.stack(map(vectors.get_vector, words))
    if do_normalize:
        m = normalize(m)
    return Basis(m, words_list=words)


# Helper functions for working with model output
def small_to_zero(x, threshold=1e-5):
    """
    Sets very small values in a numpy matrix to 0
    May modify original array
    Args:
        x (ndarray): A numpy matrix of values
        threshold(float): Cutoff threshold below which values will be set to 0
    """

    x[np.abs(x) < threshold] = 0
    return x


def word_equation(x, basis, target_word=""):
    """Given a sparse coefficient matrix embedding a word, returns a human-readable
    string containing a word vector equation

    Args:
        x (ndarray): A coefficient array, such as one learned by sparse regression
        words_vocab (list[string]): The vocabulary corresponding to the dimensions of x
    """
    if len(x.shape) > 1:
        x = x.reshape(-1)
    assert len(basis.get_words_list()) == x.size

    nonzero_indices = x.nonzero()
    nonzero_indices_list = nonzero_indices[0].tolist()
    return target_word + ' = ' + " + ".join(
        ["{:.2f} * {}".format(x[i], basis.get_words_list()[i]) for i in nonzero_indices_list])


# Functions to abstract the model fitting and word equation process

def fit_sparse_vector(target_vector, basis_vectors, alpha=1):
    model = FistaRegressor(alpha=alpha, max_iter=10)
    coefs = model.fit(basis_vectors.transpose(), target_vector)

    sparse_vector = small_to_zero(coefs.coef_)
    reconstructed_embedding = np.matmul(sparse_vector, basis_vectors)
    scaling_factor = np.linalg.norm(reconstructed_embedding, 2)
    if scaling_factor != 0:
        sparse_vector = sparse_vector / scaling_factor
    return sparse_vector.squeeze(0)


def fit_and_report(vectors, target_word, basis_size, alpha, extra_exclude=set()):
    # Preliminary regression
    basis = get_top_n_vectors(vectors, basis_size, exclude={target_word} | extra_exclude)
    original_embedding = vectors[target_word]

    sparse_embedding = fit_sparse_vector(original_embedding, basis.get_matrix(), alpha)

    reconstructed_embedding = np.matmul(sparse_embedding[None, :], basis.get_matrix())

    print(word_equation(sparse_embedding, basis, target_word=target_word))

    # Calculate representation errors
    approximation_error = np.linalg.norm(reconstructed_embedding - original_embedding, 1)
    approximation_error_L2 = np.linalg.norm(reconstructed_embedding - original_embedding, 2)
    approximation_error_cos = scipy.spatial.distance.cosine(reconstructed_embedding, original_embedding)
    print("Reconstruction error {:.2f} (L1) {:.2f} (L2) {:.2f} (cosine)".format(approximation_error,
                                                                                approximation_error_L2,
                                                                                approximation_error_cos))


def fit_ith_sparse_vector_excluding_self(vectors, basis, alpha, i):
    target_vector = vectors.syn0[i]
    target_word = vectors.index2word[i]

    basis_matrix = basis.get_matrix()

    if target_word in basis.get_words_inverse_map():
        # We need the index in the basis set
        basis_index = basis.get_words_inverse_map()[target_word]
        basis_matrix = np.copy(basis_matrix)
        basis_matrix[basis_index, :] = 0
    return fit_sparse_vector(target_vector, basis_matrix, alpha=alpha)


def fit_all_vectors(vectors, basis, alpha, reconstructed=False):
    # We want to create a new gensim.models.KeyedVectors
    # The actual vectors stored in .syn0 (and .syn0norm has the normalized version, identical here)
    # So we can copy the other data, such as vocabulary
    sparse_vectors = copy(vectors)
    num_vectors = sparse_vectors.syn0.shape[0]

    try:
        cpus = multiprocessing.cpu_count() - 2
    except NotImplementedError:
        cpus = 2  # arbitrary default
    pool = multiprocessing.Pool(processes=cpus)

    # Parallel process all vectors in vocabulary
    # .syn0 is a matrix, and we want to process each row seperatley
    # Hackiness because multiprocessing cant pickle lambda functions
    # So we use functools.partial to store the local context
    mapped_function = functools.partial(fit_ith_sparse_vector_excluding_self, vectors, basis, alpha)
    # mapped_function = functools.partial(fit_sparse_vector, basis_vectors=basis.get_matrix(), alpha=alpha)
    print("Starting multiprocessing map")
    sparse_vectors_list = list(
        tqdm.tqdm(pool.imap(mapped_function, (i for i in range(num_vectors)), chunksize=10),
                  total=num_vectors))

    if reconstructed:
        # Multiply by basis vectors
        mapped_function = functools.partial(np.matmul, b=basis.get_matrix())
        sparse_vectors_list = list(tqdm.tqdm(pool.imap(mapped_function, sparse_vectors_list)))

    # Reform into a matrix
    sparse_vectors.syn0 = np.stack(sparse_vectors_list)

    # All vectors are already normalized
    sparse_vectors.syn0norm = sparse_vectors.syn0
    # Recreate metadata
    sparse_vectors.vector_size = sparse_vectors.syn0.shape[1]
    return sparse_vectors

# vectors = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec',limit=100000)
