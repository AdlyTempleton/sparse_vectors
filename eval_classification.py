import os
import scipy
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
import numpy as np
import csv
import argparse
import string
from gensim.models import KeyedVectors
import pickle
import sparse_vectors


def add_line_to_dataset(sentence, label, dim, X, Y):
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
    Y.append(label)
    X.append(row)


def imdb_dataset(vectors, parent_dir):
    n_vectors, dim = vectors.vectors.shape

    labels = []
    X = []

    for file in os.listdir(os.path.join(parent_dir, 'pos')):
        try:
            line = open(os.path.join(parent_dir, 'pos', file), 'r').readline()
            add_line_to_dataset(line, 1, dim, X, labels)
        except UnicodeDecodeError:
            print("Skipping {}; unicode error".format(file))
    for file in os.listdir(os.path.join(parent_dir, 'neg')):
        try:
            line = open(os.path.join(parent_dir, 'neg', file), 'r').readline()
            add_line_to_dataset(line, 0, dim, X, labels)
        except UnicodeDecodeError:
            print("Skipping {}; unicode error".format(file))
    #

    X = np.stack(X, axis=0)
    # X = scipy.sparse.vstack(X).todense()
    # X = np.matmul(x, vectors.vectors)
    Y = np.array(labels, dtype=np.int)
    return X, Y


def trec_dataset(vectors, filename, labels_map=None):
    n_vectors, dim = vectors.vectors.shape
    csv_reader = csv.reader(open(filename, 'r'))
    # Skip header
    next(csv_reader)

    X, labels = [], []

    for row in csv_reader:
        question, label = row[2], row[1]
        add_line_to_dataset(question, label, dim, X, labels)
    X = np.stack(X, axis=0)

    if labels_map is None:
        unique_labels = list(set(labels))
        print(unique_labels)
        labels_map = {x: i for i, x in enumerate(unique_labels)}
    labels = [labels_map[x] for x in labels]

    Y = np.array(labels, dtype=np.int)
    return X, Y, labels_map


def train_logistic_regression(X, Y, X_test, Y_test, basis):
    lr = LogisticRegressionCV(n_jobs=4)
    lr.fit(X, Y)
    print(lr.score(X, Y))
    print(lr.score(X_test, Y_test))
    if basis is not None:
        for d in range(lr.coef_.shape[0]):
            print(sparse_vectors.word_equation(lr.coef_[d, :], basis))


def train_imdb(vectors, basis):
    print("IMDB")
    X, Y = imdb_dataset(vectors, os.path.join('aclImdb', 'train'))
    X_test, Y_test = imdb_dataset(vectors, os.path.join('aclImdb', 'test'))
    train_logistic_regression(X, Y, X_test, Y_test, basis)


def train_trec(vectors, basis):
    print("TREC")
    X, Y, labels_map = trec_dataset(vectors, 'trec.csv')
    X_test, Y_test, _ = trec_dataset(vectors, 'trec_test.csv', labels_map=labels_map)
    train_logistic_regression(X, Y, X_test, Y_test, basis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--vocab', metavar='N', type=int, default=50000,
                        help='Process the most common N words from the original vocabulary')
    parser.add_argument('--vectors-file', metavar='F', type=str, default='wiki-news-300d-1M.vec',
                        help='File to read embeddings in from')
    args = parser.parse_args()

    vectors = KeyedVectors.load_word2vec_format(args.vectors_file, limit=args.vocab)

    basis_filename = args.vectors_file.replace('.vec', '.vocab.pickle')
    basis = pickle.load(open(basis_filename, 'rb')) if os.path.exists(basis_filename) else None
    train_trec(vectors, basis)
    train_imdb(vectors, basis)
