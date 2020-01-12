import torch
import numpy as np
import torch.nn.functional as F
import sparse_vectors
import gc


def small_to_zero_pytorch(X, threshold=1e-3):
    X = X.detach()
    X[X.abs() < threshold] = 0
    return X


class PyruSparseRegressor():
    def __init__(self, alpha, n_basis, n_target):
        self.alpha = alpha
        self.n_target = n_target
        self.coef = torch.nn.Parameter(torch.zeros(n_target, n_basis, device=torch.device('cuda')))

    def fit(self, X, Y, lr, num_iter, excluded=None, return_reconstructed=False):
        """
        X: (n_basis, dim)
        Y: (n_target, dim)
        """
        torch.cuda.empty_cache()
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, device=torch.device('cuda'))
        if isinstance(Y, np.ndarray):
            Y = torch.tensor(Y, device=torch.device('cuda'))

        # E

        self.coef.data.normal_()
        optimizer = torch.optim.Adam([self.coef], lr=lr)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda i: 1 - (i / num_iter))
        # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lambda  i : 1)

        for i in range(num_iter):

            Y_prime = self.coef.mm(X)
            representation_loss = (Y - Y_prime).pow(2).sum()
            penalty_loss = (self.alpha * self.coef.abs().sum())
            loss = representation_loss + penalty_loss

            loss.backward()
            with torch.no_grad():
                # adjusted_lr = lr * min(1, 10/1)
                # self.coef.sub_(adjusted_lr * self.coef.grad)
                optimizer.step()
                scheduler.step()
                if excluded is not None:
                    self.coef[excluded[:, 0], excluded[:, 1]] = 0
                optimizer.zero_grad()
            if i % 100 == 0:
                print("{}: {:.6f} ({:.4f}) ({}) ({:.1f})".format(i, loss.item() / self.n_target,
                                                                 self.coef.sum(dim=-1).mean().item(),
                                                                 (self.coef.sum(dim=1) != 0).sum(), float(
                        (self.coef.cpu().abs() > 1e-3).sum()) / self.n_target))

        # Free memory for the next step
        del Y_prime
        del optimizer
        self.coef.grad = None
        self.coef.data.grad = None
        torch.cuda.empty_cache()
        # Reconstruct and rescale
        # Note that the scaling has to be normal not in the space of sparse embeddings but in the space of reconstructed embeddings
        # TODO: Are these the same?
        sparse = small_to_zero_pytorch(self.coef)
        reconstructed = sparse.mm(X)
        scaling_factors = reconstructed.pow(2).sum(dim=-1, keepdim=True)
        sparse, reconstructed = sparse / scaling_factors, reconstructed / scaling_factors
        return sparse.detach().cpu().numpy() if not return_reconstructed else reconstructed.detach().cpu().numpy()
