import multiprocessing
import argparse
from gensim.models import KeyedVectors
import argparse
import multiprocessing
from sparse_vectors import *
import pickle

from gensim.models import KeyedVectors

def run_sparse_vectors(vectors, args):
    vectors = center_normalize_vectors(vectors)
    basis = (get_top_n_vectors(vectors, args.basis if args.basis_filter is None else 50000, exclude={}))
    if args.syntactic is not None:
        basis_syn = get_pca_basis(vectors).merge(get_pos_basis(vectors)).merge(
            get_syntactic_basis(vectors, args.syntactic))
        if args.sparse_syn:
            basis_syn.n_syntactic = 0
        basis = basis.subtract_projection_and_merge(basis_syn.orthogonalize())
        if args.basis_filter is not None:
            basis_filter = pickle.load(open(args.basis_filter, 'rb'))
            if args.basis < len(basis_filter):
                basis_filter = basis_filter[:args.basis]
            basis = basis.select_words(basis_filter)
    return basis, fit_all_vectors(vectors, basis, alpha=args.alpha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--vocab', metavar='N', type=int, default=200000,
                        help='Process the most common N words from the original vocabulary')
    parser.add_argument('--basis', metavar='B', type=int, default=20000,
                        help='Number of words to use as the initial basis vocabulary, before filtering')
    parser.add_argument('--basis-filter', metavar='BF', type=str, default=None,
                        help='Filename which contains a filter for a basis')
    parser.add_argument('--alpha', metavar='A', type=float, default=1,
                        help='Number of words to use as the initial basis vocabulary, before filtering')
    parser.add_argument('--syntactic', type=str, default=None,
                        help='If true, add syntactic vectors from the given filename')

    parser.add_argument('--sparse-syn', action='store_true', default=False,
                        help='If true, treat syntactic basis with the standard sparse coding')
    parser.add_argument('--input', metavar='I', type=str, default='wiki-news-300d-1M.vec',
                        help='File to read dense embeddings from')
    parser.add_argument('--output', metavar='O', type=str, default=None,
                        help='File to which to save sparse embeddings')
    args = parser.parse_args()

    output_file = args.output if args.output is not None else 'sparse_vectors{}{}-{}-{}-{}'.format(
        '-' + args.syntactic if args.syntactic is not None else '',
        '-sparsesyn' if args.sparse_syn is not None else '',
        args.vocab, args.basis, args.alpha)

    vectors = KeyedVectors.load_word2vec_format(args.input, limit=args.vocab)
    basis, (sparse_vectors, reconstructed_vectors) = run_sparse_vectors(vectors, args)
    sparse_vectors.save_word2vec_format(output_file + '.vec')

    reconstructed_vectors.save_word2vec_format(output_file + '-reconstructed.vec')
    pickle.dump(basis, open(output_file + '.vocab.pickle', 'wb'))
