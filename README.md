# Code for the paper "Inherently Interpretable Sparse Word Embeddings through Sparse Coding"

The main entrypoint into this library is the run_sparse_vectors.py file. This is a script which will take as input dense vectors (in gensim format) and output sparse vectors (in gensim format) transformed as described in the paper. For example, to generate the vectors described in the paper, use a command like:

  python run_sparse_vectors.py --input wiki-news-300d-1M.vec --vocab 50000 --basis 3000 --alpha .1 --syntactic syntactic.txt --basis-filter basis_guided_filtered.pickle

Dependencies are listed in conda.yml
