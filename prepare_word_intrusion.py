import random
import numpy as np
from gensim.models import KeyedVectors
import csv
import pickle
from sparse_vectors import *

random.seed(1234)

isascii = lambda s: len(s) == len(s.encode())


def prepare_word_intrusion(csv_writer, vectors, dims_to_sample, code, hints=None):
    """
    Writes a set of word intrusion questions from a single vector set
    """
    n_vectors, dims = vectors.vectors.shape
    sampled_dims = random.sample(range(dims), dims_to_sample)
    for d in sampled_dims:
        values_on_dim = vectors.vectors[:, d]
        ranks_on_dim = np.argsort(values_on_dim)[::-1]
        # We select the top 4
        correct_indices = ranks_on_dim[:4].tolist()
        # And one from the bottom 50%
        intruder = ranks_on_dim[random.randint(n_vectors // 2, n_vectors)]
        combined_indices = correct_indices + [intruder]
        random.shuffle(combined_indices)
        combined_words = [vectors.index2word[i] for i in combined_indices]
        correct_answer = combined_indices.index(intruder)
        csv_writer.writerow([code] + combined_words + [correct_answer] + [hints[d] if hints is not None else ''])


if __name__ == "__main__":
    limit = 20000
    isascii = lambda s: len(s) == len(s.encode())
    valid_word_filter = lambda s: s.islower() and s.isalpha() and isascii(s) and len(s) > 1
    vectors_fasttext = KeyedVectorsPlus(KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', limit=limit)).filter(
        valid_word_filter)
    vectors_sparse = KeyedVectorsPlus(
        KeyedVectors.load_word2vec_format('data/sparse_vectors-basis_filtered_guided.pickle-30000-3000-0.1.vec',
                                          limit=limit)).filter(valid_word_filter)
    vectors_sparse_basis = pickle.load(
        open('data/sparse_vectors-basis_filtered_guided.pickle-30000-3000-0.1.vocab.pickle', 'rb'))
    vectors_faraqui = KeyedVectorsPlus(
        KeyedVectors.load_word2vec_format('data/faraqui-sparse-coding-0.75.vec', limit=limit)).filter(valid_word_filter)
    with open('wordintrusion.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        prepare_word_intrusion(csv_writer, vectors_fasttext, 270, 'FASTTEXT')
        prepare_word_intrusion(csv_writer, vectors_sparse, 270, 'OURS-SYNSPARSE')
        prepare_word_intrusion(csv_writer, vectors_sparse, 270, 'OURS-SYNSPARSE-HINTS', vectors_sparse_basis.words_list)
        prepare_word_intrusion(csv_writer, vectors_sparse, 270, 'FARAQUI')
    # Now we rewrite into survey monkey format
    with open('wordintrusion.csv', 'r') as csvfile:
        with open('wordintrusion-final.csv', 'w') as outfile:
            lines = list(csvfile.readlines())
            header = ','.join(['{0}label,{0}w1,{0}w2,{0}w3,{0}w4,{0}w5,{0}answer,{0}hint'.format(i) for i in range(9)])
            outfile.write(header)
            random.shuffle(lines)
            for i, line in enumerate(lines):
                outfile.write('\n' if i % 9 == 0 else ',')
                outfile.write(line.strip())
