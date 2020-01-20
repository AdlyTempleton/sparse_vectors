import os
import scipy
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import numpy as np
import argparse
import string
from gensim.models import KeyedVectors


def imdb_dataset(vectors, parent_dir):
    n_vectors, dim = vectors.vectors.shape

    labels = []
    X = []

    def add_line(sentence, label):
        sentence = sentence.replace("<br />", "")
        sentence = sentence.translate(str.maketrans('', '', string.punctuation)).lower()
        # row = scipy.sparse.dok_matrix((1,n_vectors))
        row = np.zeros((dim,))
        n = 0
        for word in [x.lower() for x in sentence.split()]:
            if word in vectors.vocab:
                row += vectors.get_vector(word)
                n += 1
        if n > 0:
            row = row / n
        labels.append(label)
        X.append(row)

    for file in os.listdir(os.path.join(parent_dir, 'pos')):
        try:
            line = open(os.path.join(parent_dir, 'pos', file), 'r').readline()
            add_line(line, 1)
        except UnicodeDecodeError:
            print("Skipping {}; unicode error".format(file))
    for file in os.listdir(os.path.join(parent_dir, 'neg')):
        try:
            line = open(os.path.join(parent_dir, 'neg', file), 'r').readline()
            add_line(line, 0)
        except UnicodeDecodeError:
            print("Skipping {}; unicode error".format(file))
    #

    X = np.stack(X, axis=0)
    # X = scipy.sparse.vstack(X).todense()
    # X = np.matmul(x, vectors.vectors)
    Y = np.array(labels, dtype=np.int)
    return X, Y


def train_imdb(vectors):
    parent_dir = 'aclImdb'

    X, Y = imdb_dataset(vectors, os.path.join(parent_dir, 'train'))

    lr = LogisticRegression(C=.1)
    lr.fit(X, Y)
    X_test, Y_test = imdb_dataset(vectors, os.path.join(parent_dir, 'test'))
    print(lr.score(X, Y))
    print(lr.score(X_test, Y_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--vocab', metavar='N', type=int, default=50000,
                        help='Process the most common N words from the original vocabulary')
    parser.add_argument('--vectors-file', metavar='F', type=str, default='wiki-news-300d-1M.vec',
                        help='File to read embeddings in from')
    args = parser.parse_args()

    vectors = KeyedVectors.load_word2vec_format(args.vectors_file, limit=args.vocab)
    train_imdb(vectors)
