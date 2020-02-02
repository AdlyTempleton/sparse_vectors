import gensim
import numpy as np
import cupy as cp
import gc
from gensim.models import KeyedVectors
import random
from copy import copy
import multiprocessing
from collections import OrderedDict
from sklearn.linear_model import LinearRegression
import sklearn
import functools
import lightning
import sys
from lightning.regression import FistaRegressor
import spacy
import math
import scipy
import tqdm

from sklearn.decomposition import PCA
import en_core_web_sm

# Load pretrained fastText vectors

spacy_nlp = en_core_web_sm.load(disable=['parser', 'ner'])


def xp(x):
    return cp.get_array_module(x)


def normalize(x):
    return x / xp(x).linalg.norm(x, axis=1, keepdims=True)


class KeyedVectorsPlus(KeyedVectors):
    """Implements some additional methods and comparibility fixes from KeyedVectors"""

    def __init__(self, a):

        super(KeyedVectorsPlus, self).__init__(vector_size=a.vector_size)
        self.vectors = a.vectors
        self.vectors_norm = a.vectors_norm
        self.vocab = a.vocab
        self.index2entity = a.index2word
        self.index2entity = a.index2entity

    def word_vec(self, word, use_norm=False):
        """Overwrite to remove cupy-breaking setflags call"""
        if word in self.vocab:
            if use_norm:
                result = self.vectors_norm[self.vocab[word].index]
            else:
                result = self.vectors[self.vocab[word].index]

            # result.setflags(write=False)
            return result
        else:
            raise KeyError("word '%s' not in vocabulary" % word)

    def like(self, vector_matrix, do_norm=True, do_cupy=False):
        """Creates a new gensim.KeyedVectors with the same vocabulary as old_vectors but with the embedding matrix of vector_matrix"""
        new_vectors = copy(self)
        new_vectors.vectors = cp.asarray(vector_matrix) if do_cupy else vector_matrix
        vectors_norm = normalize(vector_matrix) if do_norm else vector_matrix
        new_vectors.vectors_norm = cp.asarray(vectors_norm) if do_cupy else vectors_norm
        new_vectors.vector_size = new_vectors.vectors.shape[1]
        return new_vectors

    def to_cupy(self):
        return self.like(self.vectors, do_cupy=True)

    def filter(self, filter):
        """Returns a vectors object containing only lowercase entries from the original object"""
        filtered = copy(self)
        filtered_indices, filtered_words = zip(*[(i, s) for i, s in enumerate(self.index2word) if filter(s)])
        filtered.index2word = filtered_words

        filtered.vectors = self.vectors[list(filtered_indices), :]
        if self.vectors_norm is not None:
            filtered.vectors_norm = self.vectors_norm[list(filtered_indices), :]
        return filtered


class Basis:
    def __init__(self, matrix, words_list=None, gensim_vocab=None, n_syntactic=0):
        """We can construct from either a list of words or a gensim Vocab object, but not both
        matrix is a ndarray"""
        # The number of basis vectors which make up the 'syntactic' basis. These are always at the front
        self.n_syntactic = n_syntactic
        assert isinstance(matrix, cp.core.core.ndarray) or isinstance(matrix, np.ndarray)
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

    def to_numpy(self):
        if isinstance(self.matrix, np.ndarray):
            return self
        else:
            return Basis(matrix=self.matrix.get(), words_list=self.words_list, n_syntactic=self.n_syntactic)

    def merge(self, other):
        # If this is a pure merge, we have an easy case
        if len(self) == 0:
            return other
        if len(other) == 0:
            return self
        if not self.is_mixed_syntactic_semantic() and not other.is_mixed_syntactic_semantic():
            # However, we do want to make sure we get the correct order
            if self.n_syntactic == 0 and other.n_syntactic != 0:
                return other.merge(self)
            else:
                return Basis(matrix=np.concatenate((self.matrix, other.matrix)),
                             words_list=self.get_words_list() + other.get_words_list(),
                             n_syntactic=self.n_syntactic + other.n_syntactic)
        else:
            # Otherwies, we do a bit of 'recursion', merging the semantic and syntactic components
            # And then merging together
            return (self.get_syntactic().merge(other.get_syntactic())).merge(
                self.get_semantic().merge(other.get_semantic()))

    def subtract_projection_and_merge(self, other):
        """Subtracts the projection onto other from self, and merges the two bases
        Used to combine a syntactic and a semantic basis"""
        _, residuals = fit_to_basis(other.matrix, self.matrix)
        return Basis(normalize(residuals), words_list=self.words_list, n_syntactic=self.n_syntactic).merge(other)

    def __len__(self):
        return len(self.words_list)

    def is_mixed_syntactic_semantic(self):
        return self.n_syntactic != 0 and self.n_syntactic != len(self)

    def exclude(self, excluded):
        excluded_indices = [i for i, x in enumerate(self.get_words_list()) if x in excluded]
        new_matrix = np.delete(self.matrix, excluded_indices, axis=0)
        new_words_list = [x for x in self.words_list if x not in excluded]
        return Basis(matrix=new_matrix, words_list=new_words_list, n_syntactic=self.n_syntactic)

    def slice(self, start, end):
        return Basis(matrix=self.matrix[start:end, :], words_list=self.words_list[start:end],
                     n_syntactic=max(0, min(self.n_syntactic, end) - start))

    def get_syntactic(self):
        return self.slice(0, self.n_syntactic)

    def get_semantic(self):
        return self.slice(self.n_syntactic, len(self.words_list))

    def orthogonalize(self):
        """Iteratively creates an orthonormal basis via the gram schmidt process"""
        orthonormal_matrix = normalize(self.matrix[np.newaxis, 0, :])
        for i in range(1, len(self)):
            vector = self.matrix[np.newaxis, i, :]
            _, residual = fit_to_basis(orthonormal_matrix, vector)
            orthonormal_matrix = np.concatenate((orthonormal_matrix, normalize(residual)), axis=0)
        return Basis(orthonormal_matrix, words_list=self.words_list, n_syntactic=self.n_syntactic)

    def select(self, indices):
        return Basis(words_list=[self.words_list[i] for i in indices], matrix=self.matrix[indices, :],
                     n_syntactic=len([x for x in indices if x < self.n_syntactic]))

    def select_words(self, words_list, ignore_missing=False):
        if ignore_missing:
            our_words_set = set(self.words_list)
            words_list = [x for x in words_list if x in our_words_set]
        indices = [self.words_inv[w] for w in words_list]
        print(indices)
        return self.select(indices)

    def sample(self, p):
        if self.n_syntactic > 0:
            return self.get_syntactic().merge(self.get_semantic().sample(p))
        else:
            k = int(p * len(self))
            indices = random.sample(range(len(self)), k)
            return self.select(indices)

    def filter_maximum_distinct(self, n):
        basis = self.get_syntactic()
        while len(basis) < n:
            i = np.argmin(np.max(np.matmul(self.matrix, np.transpose(basis.matrix)), axis=1), axis=0)
            basis = basis.merge(self.slice(i, i + 1))
        return basis

    def filter_vocab_guided(self, vectors, n, verbose=False, use_cupy=True):
        basis = self.get_syntactic()
        weights_by_vector = 1 - np.max(np.abs(np.matmul(vectors.vectors, np.transpose(basis.matrix))), axis=1)
        basis_vector_sim = np.abs(np.matmul(self.matrix, np.transpose(vectors.vectors)))
        basis_vector_sim[0:len(basis), :] = 0

        if use_cupy:
            weights_by_vector, basis_vector_sim = cp.asarray(weights_by_vector, dtype=cp.float16), cp.asarray(
                basis_vector_sim, dtype=cp.float16)

        xp = cp if use_cupy else np

        while len(basis) < n:
            if verbose:
                print(len(basis))
            weighted_similarities = xp.mean(weights_by_vector * basis_vector_sim, axis=1)
            i = int(xp.argmax(weighted_similarities))
            # Add to new basis
            basis = basis.merge(self.slice(i, i + 1).to_numpy())
            # Decrease appropriate weights
            weights_by_vector = xp.minimum(1 - xp.abs(basis_vector_sim[i, :]), weights_by_vector)
            # Exclude this basis from being picked
            basis_vector_sim[i, :] = 0
        del weighted_similarities
        del weights_by_vector
        del basis_vector_sim
        return basis

    def filter_good_words(self):
        filter_fn = lambda w: spacy_nlp(w)[0].pos_ in {'NOUN', 'VERB', 'ADJ'} and w in spacy_nlp.vocab and w.islower()
        return self.get_syntactic().merge(self.select([i for i, w in enumerate(self.words_list) if filter_fn(w)]))

    def make_semantic(self):
        b = copy(self)
        b.n_syntactic = 0




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
            total_matrix = np.stack(vector_diffs, axis=0)
            basis_vectors.append(np.mean(total_matrix, axis=0))
            # We want to take the elementwise minimum absolute value along axis 0
            # basis_vectors.append(total_matrix[np.argmin(np.abs(total_matrix), axis=0),np.arange(total_matrix.shape[1])])

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
                    vector_diffs.append(vec_b - vec_a)
                except KeyError as e:
                    print(e)
                    print("Skipping pair {},{}".format(a, b))
    finish_section()
    return Basis(np.stack(basis_vectors, axis=0), words_list=basis_words_list,
                 n_syntactic=len(basis_vectors)).orthogonalize()


def center_normalize_vectors(vectors):
    np = xp(vectors.vectors)
    return vectors.like(np.subtract(vectors.vectors, np.mean(vectors.vectors, axis=0, keepdims=True)))


def get_pos_basis(vectors):
    pos = list(map(lambda word: spacy_nlp(word)[0].pos_, vectors.index2word))
    # Get unique parts of speech and create an inverse index
    # Order matters herer because it becomes the order of orthogonalization
    pos_groups = OrderedDict()
    for p in ['NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'NUM']:
        pos_groups[p] = []
    for i, x in enumerate(pos):
        if x in pos_groups:
            pos_groups[x].append(i)
    # zip(*x) = unzup
    pos_names_list, pos_indices_list = tuple(zip(*pos_groups.items()))
    pos_names_list = ['<<<POS-{}>>>'.format(x) for x in pos_names_list]
    pos_mean_vectors = np.stack(
        [np.mean(np.stack([vectors.vectors[i] for i in indices], axis=0), axis=0) for indices in pos_indices_list])
    return Basis(pos_mean_vectors, words_list=pos_names_list, n_syntactic=len(pos_names_list))


def get_capitalization_basis(vectors):
    cap_vector = np.mean(vectors.filter(lambda w: w[0].isupper()).vectors, axis=0, keepdims=True)
    return Basis(cap_vector, words_list=['<<<CAPITALIZATION>>>'], n_syntactic=1)


def get_pca_basis(vectors):
    pca = PCA(n_components=1)
    pca.fit(vectors.vectors)
    return Basis(
        matrix=normalize(pca.components_),
        words_list=['<<<C0>>>'], n_syntactic=1)


def get_combined_syntactic_basis(vectors, syntactic_filename='syntactic.txt'):
    return get_pca_basis(vectors).merge(get_capitalization_basis(vectors)) \
        .merge(get_pos_basis(vectors)) \
        .merge(get_syntactic_basis(vectors, filename=syntactic_filename)) \
        .orthogonalize()


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
    m = np.stack(list(map(vectors.get_vector, words)))
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
    terms = ["{:.2f} * {}".format(x[i], basis.get_words_list()[i]) for i in nonzero_indices_list]
    # sort terms by decreasing weight
    terms = [x[1] for x in sorted(zip([abs(x[i]) for i in nonzero_indices_list], terms), reverse=True)]
    return target_word + ' = ' + " + ".join(terms)


# Functions to abstract the model fitting and word equation process

def fit_sparse_vector(target_vector, basis_vectors, alpha=1):
    model = FistaRegressor(alpha=alpha, max_iter=100, max_steps=30)
    coefs = model.fit(basis_vectors.transpose(), target_vector)

    sparse_vector = small_to_zero(coefs.coef_)
    reconstructed_embedding = np.matmul(sparse_vector, basis_vectors)
    scaling_factor = np.linalg.norm(reconstructed_embedding, 2)
    if scaling_factor != 0:
        sparse_vector = sparse_vector / scaling_factor
    return sparse_vector.squeeze(0)


def fit_and_report(vectors, target_word, basis, alpha, extra_exclude=set(), sparse_syn=True):
    """Designed to be called primarily from a notebook"""
    # Preliminary regression
    basis = basis.exclude({target_word} | extra_exclude)

    syntactic_loadings = None

    original_basis = basis
    original_embedding = vectors[target_word]
    vector_to_embed = original_embedding

    if basis.n_syntactic > 0 and not sparse_syn:
        syntactic_loadings, vector_to_embed = fit_all_syntactic(original_embedding, basis)
        basis = basis.get_semantic()

    if not isinstance(alpha, list):
        alpha = [alpha]

    r_cos, r_nonzero = [], []
    for a in alpha:
        sparse_embedding = fit_sparse_vector(vector_to_embed, basis.get_matrix(), a)
        if syntactic_loadings is not None:
            sparse_embedding = np.concatenate((syntactic_loadings, sparse_embedding),
                                              axis=0)
        reconstructed_embedding = np.matmul(sparse_embedding[np.newaxis, :], original_basis.get_matrix())

        # Calculate representation errors

        approximation_error = np.linalg.norm(reconstructed_embedding - original_embedding, 1)
        approximation_error_L2 = np.linalg.norm(reconstructed_embedding - original_embedding, 2)
        approximation_error_cos = scipy.spatial.distance.cosine(reconstructed_embedding, original_embedding)
        r_cos.append(approximation_error_cos)
        r_nonzero.append(np.count_nonzero(sparse_embedding))

        if len(alpha) == 1:
            print(word_equation(sparse_embedding, original_basis, target_word=target_word))
            print("Reconstruction error {:.2f} (L1) {:.2f} (L2) {:.2f} (cosine)".format(approximation_error,
                                                                                        approximation_error_L2,
                                                                                        approximation_error_cos))
    return r_cos, r_nonzero


def fit_ith_sparse_vector_excluding_self(vectors, basis, alpha, i):
    target_vector = vectors.vectors[i]
    target_word = vectors.index2word[i]

    basis_matrix = basis.get_matrix()

    if target_word in basis.get_words_inverse_map():
        # We need the index in the basis set
        basis_index = basis.get_words_inverse_map()[target_word]
        basis_matrix = np.copy(basis_matrix)
        # We zero out the row and restore it later
        saved_vector = basis_matrix[basis_index, :]
        basis_matrix[basis_index, :] = 0
    r = fit_sparse_vector(target_vector, basis_matrix, alpha=alpha)
    if target_word in basis.get_words_inverse_map():
        basis_matrix[basis_index, :] = saved_vector
    return r


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



def fit_to_basis(basis_matrix, vectors_matrix):
    """Fits a single vector to a syntactic basis"""
    loadings = np.matmul(vectors_matrix, np.transpose(basis_matrix))
    # Projection has size (num_vectors, num_basis)
    # We expand to (num_vectors, num_basis, d)
    num_vectors, d = vectors_matrix.shape
    projection = np.repeat(loadings[:, :, np.newaxis], d, axis=2)
    # Now we repeat the basis for each vector
    basis_matrix = np.repeat(basis_matrix[np.newaxis, :, :], num_vectors, axis=0)
    # elementwise multiply and sum along num_basis
    approximation = np.sum(basis_matrix * projection, axis=1)
    residuals = vectors_matrix - approximation
    return loadings, residuals


def fit_all_syntactic(vectors, basis):
    """Fits and removes projection along a dense syntactic basis"""
    return fit_to_basis(basis.get_syntactic().get_matrix(), vectors.vectors)


def fit_all_vectors(vectors, basis, alpha, reconstructed=False):
    # We want to create a new gensim.models.KeyedVectors
    # The actual vectors stored in .syn0 (and .syn0norm has the normalized version, identical here)
    # So we can copy the other data, such as vocabulary
    num_vectors = vectors.syn0.shape[0]

    # Fit syntactic basis
    syntactic_loadings = None
    if basis.n_syntactic > 0:
        syntactic_loadings, residuals = fit_all_syntactic(vectors, basis)
        vectors = vectors.like(residuals)

        original_basis = basis
        basis = basis.get_semantic()

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
    sparse_vectors_matrix = np.stack(sparse_vectors_list)
    if syntactic_loadings is not None:
        sparse_vectors_matrix = np.concatenate((syntactic_loadings, sparse_vectors_matrix), axis=1)
        basis = original_basis
    sparse_vectors = vectors.like(sparse_vectors_matrix)
    # Reconstruct the approximation to the original dense vectors represented by the sparse vectors
    print("Reconstructing")
    reconstructed = vectors.like(np.matmul(sparse_vectors.syn0, basis.get_matrix()))
    return sparse_vectors, reconstructed

# vectors = KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec',limit=100000)
