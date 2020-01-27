import random
import numpy as np
from gensim.models import KeyedVectors
import csv
from sparse_vectors import *

random.seed(1234)

isascii = lambda s: len(s) == len(s.encode())


def prepare_word_intrusion(csv_writer, vectors, dims_to_sample, code):
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
        csv_writer.writerow([code] + combined_words + [correct_answer])


if __name__ == "__main__":
    limit = 20000
    isascii = lambda s: len(s) == len(s.encode())
    valid_word_filter = lambda s: s.islower() and s.isalpha() and isascii(s) and len(s) > 1
    vectors_fasttext = keyedvector_filter(KeyedVectors.load_word2vec_format('wiki-news-300d-1M.vec', limit=limit),
                                          valid_word_filter)
    vectors_sparse = keyedvector_filter(
        KeyedVectors.load_word2vec_format('sparse_vectors-synsparse-guided-30000-3000-.15.vec', limit=limit),
        valid_word_filter)
    with open('wordintrusion.csv', 'w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        prepare_word_intrusion(csv_writer, vectors_fasttext, 100, 'FASTTEXT')
        prepare_word_intrusion(csv_writer, vectors_sparse, 100, 'OURS-SYNSPARSE')
    # Now we rewrite into survey monkey format
    with open('wordintrusion.csv', 'r') as csvfile:
        with open('wordintrusion.key', 'w') as keyfile:
            with open('wordintrusion.txt', 'w') as outfile:
                lines = list(csvfile.readlines())
                random.shuffle(lines)
                for rowtext in lines:
                    row = rowtext.split(',')
                    outfile.write('Pick the word which is most unlike the rest of the words\n')
                    answers = row[1:-1]
                    outfile.write('\n'.join(answers))
                    outfile.write('\n\n')
                    keyfile.write('{},{}\n'.format(row[0], row[-1]))
