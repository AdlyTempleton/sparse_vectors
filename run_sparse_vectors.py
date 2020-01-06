import multiprocessing
import argparse
from gensim.models import KeyedVectors
import argparse
import multiprocessing
from sparse_vectors import *
import pickle

from gensim.models import KeyedVectors


def run_sparse_vectors(basis_size, alpha, reconstructed):
    basis = get_top_n_vectors(vectors, basis_size, exclude={})
    return basis, fit_all_vectors_pytorch(vectors, basis, alpha=alpha, reconstructed=reconstructed)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--vocab', metavar='N', type=int, default=200000,
                        help='Process the most common N words from the original vocabulary')
    parser.add_argument('--basis', metavar='B', type=int, default=20000,
                        help='Number of words to use as the initial basis vocabulary, before filtering')
    parser.add_argument('--alpha', metavar='A', type=float, default=1,
                        help='Number of words to use as the initial basis vocabulary, before filtering')
    parser.add_argument('--reconstruct', action='store_true', default=False,
                        help='If true, output the approxmated dense vectors instead of the sparse vectors')
    parser.add_argument('--input', metavar='I', type=str, default='wiki-news-300d-1M.vec',
                        help='File to read dense embeddings from')
    parser.add_argument('--output', metavar='O', type=str, default=None,
                        help='File to which to save sparse embeddings')
    args = parser.parse_args()

    output_file = args.output if args.output is not None else 'sparse_vectors{}-{}-{}-{}.'.format(
        '-reconstructed' if args.reconstruct else '', args.vocab, args.basis, args.alpha)

    vectors = KeyedVectors.load_word2vec_format(args.input, limit=args.vocab)
    basis, sparse_vectors = run_sparse_vectors(args.basis, args.alpha, args.reconstruct)
    sparse_vectors.save_word2vec_format(output_file + 'vec')
    pickle.dump(basis, open(output_file + 'vocab.pickle', 'wb'))
