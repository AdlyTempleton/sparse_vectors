import gensim
import numpy as np
import gc
from gensim.models import KeyedVectors
from copy import copy
import multiprocessing
import sklearn
import functools
import lightning
import sys
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

    def merge(self, other):
        return Basis(matrix=np.concatenate((self.matrix, other.matrix)),
                     words_list=self.get_words_list() + other.get_words_list())


def get_syntactic_basis(vectors, filename='syntactic.txt'):
    # Files are formatted with headers, starting with <, and then a list of pairs of words (command-seperated)
    with open(filename, 'r') as file:
        # Keep track of the previous sections that we have parsed
        basis_words_list = []
        basis_vectors = []
        # name of current section we are parsing
        name = None
        # Vectors in current section we are parsing
        vector_diffs = []

        def finish_section():
            basis_words_list.append(name)
            basis_vectors.append(np.mean(np.stack(vector_diffs), axis=0))

        for line in file:
            if line[0] == '<':
                # If this is not the first section
                if name is not None:
                    finish_section()
                name = line
            else:
                try:
                    a, b = line.strip().split(',')
                    vec_a, vec_b = vectors.get_vector(a), vectors.get_vector(b)
                    vector_diffs.append(vec_a - vec_b)
                except KeyError as e:
                    print(e)
                    print("Skipping pair {},{}".format(a, b))
    finish_section()
    return Basis(np.stack(basis_vectors, axis=0), words_list=basis_words_list)
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
def small_to_zero(x, threshold=1e-3):
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


def fit_chunk_of_vectors(vectors, basis, alpha, chunk):
    """Calls fit_ith_vector_excluding_self for each i in the range(chunk), and returns the output as a list
    Used to prevent expensive serialization for every call when multiprocessing"""
    start, end = chunk
    r = (fit_ith_sparse_vector_excluding_self(vectors, basis, alpha, i) for i in range(start, end))
    # If start == 0 then dispaly a progress bar
    return list(r) if start != 0 else list(tqdm.tqdm(r, total=end))


def get_chunks(n, m):
    """Returns chunk boundary tuples to divide a list of size n into m almost equal sized ranges"""
    # First, we calculate the sizes
    chunk_sizes = [n // m] * m
    remainder = n - sum(chunk_sizes)
    chunk_sizes = [x + (1 if i < remainder else 0) for i, x in enumerate(chunk_sizes)]

    # Now we take partial sums to get chunk boundaries
    cum_sum = 0
    chunk_boundaries = []
    for x in chunk_sizes:
        cum_sum += x
        chunk_boundaries.append(cum_sum)

    # Now we stagger these boundaries and zip them together
    # Note that zip cuts off the dangling half-chunk starting at n
    return zip([0] + chunk_boundaries, chunk_boundaries)


def fit_all_vectors(vectors, basis, alpha, reconstructed=False):
    # We want to create a new gensim.models.KeyedVectors
    # The actual vectors stored in .syn0 (and .syn0norm has the normalized version, identical here)
    # So we can copy the other data, such as vocabulary
    sparse_vectors = copy(vectors)
    num_vectors = sparse_vectors.syn0.shape[0]

    try:
        cpus = min(30, multiprocessing.cpu_count() - 2)
    except NotImplementedError:
        cpus = 2  # arbitrary default
    pool = multiprocessing.Pool(processes=cpus)

    # Parallel process all vectors in vocabulary
    # .syn0 is a matrix, and we want to process each row seperatley
    # Hackiness because multiprocessing cant pickle lambda functions
    # So we use functools.partial to store the local context
    mapped_function = functools.partial(fit_chunk_of_vectors, vectors, basis, alpha)
    print("Starting multiprocessing map with {} CPUs".format(cpus))
    sparse_vectors_list = pool.imap(mapped_function, get_chunks(num_vectors, cpus), chunksize=1)

    # Flatten the list-of-lists
    sparse_vectors_list = sum(sparse_vectors_list, [])

    # Reform into a matrix
    sparse_vectors.syn0 = np.stack(sparse_vectors_list)

    # All vectors are already normalized
    sparse_vectors.syn0norm = sparse_vectors.syn0
    # Recreate metadata
    sparse_vectors.vector_size = sparse_vectors.syn0.shape[1]

    # Reconstruct the approximation to the original dense vectors represented by the sparse vectors
    print("Reconstructing")
    reconstructed = copy(sparse_vectors)
    reconstructed.syn0 = np.matmul(sparse_vectors.syn0, basis.get_matrix())
    reconstructed.syn0_norm = reconstructed.syn0
    reconstructed.vector_size = reconstructed.syn0.shape[1]
    return sparse_vectors, reconstructed

# vectors = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec',limit=100000)
