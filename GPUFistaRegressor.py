from lightning.regression import FistaRegressor
import torch
import numpy as np
from lightning.datasets import get_dataset


class GPUFistaRegressor(FistaRegressor):

    def _get_regularized_objective(self, df, y, loss, penalty, coef):
        loss = self.C * (df - y).pow(2).sum()
        loss = loss + self.alpha * (coef.abs().sum())
        return loss

    def _get_quad_approx(self, coefa, coefb, objb, gradb, L, penalty):
        approx = objb
        diff = coefa - coefb
        approx += (diff * gradb).sum()
        approx += L / 2 * diff.pow(2).sum()
        approx += self.alpha * coefa.abs().sum()
        return approx

    def _fit(self, X, y, n_vectors):
        X, y = torch.tensor(X, device=torch.cuda), torch.tensor(y, device=torch.cuda)
        n_samples, n_features = X.size()

        loss = self._get_loss()
        penalty = self._get_penalty()

        # ds = get_dataset(X)

        df = torch.zeros((n_samples, n_vectors), dtype=torch.float64, device=torch.cuda)
        coef = torch.zeros((n_vectors, n_features), dtype=torch.float64, device=torch.cuda)
        coefx = coef

        G = torch.zeros((n_vectors, n_features), dtype=np.float64, device=torch.cuda)

        # Do not bother to compute the Lipschitz constant (expensive).
        L = 1.0

        t = 1.0

        for it in range(self.max_iter):
            if self.verbose >= 1:
                print("Iter", it + 1)

            # Save current values
            t_old = t
            coefx_old = coefx

            # Gradient
            G.fill_(0.0)
            # Rewrite this line to use pytorch
            loss.gradient(df, ds, y, G)
            for i in range(n_samples):
                for k in range(n_vectors):
                    residual = y[i, k] - df[i, k
                    for jj in range(n_nz):
                        j = indices[jj]
                    G[k, j] -= residual * data[jj]
                    G *= self.C

                    # Line search
                    if self.max_steps > 0:
                        objb = self._get_objective(df, y, loss)

                    for tt in xrange(self.max_steps):
                    # Solve
                        coefx = coef - G / L
                    coefx = penalty.projection(coefx, self.alpha, L)

                    dfx = safe_sparse_dot(X, coefx.T)
                    obj = self._get_regularized_objective(dfx, y, loss, penalty,
                                                          coefx)
                    approx = self._get_quad_approx(coefx, coef, objb, G, L, penalty)

                    accepted = obj <= approx

                    # Sufficient decrease condition
                    if accepted:
                        if
                    self.verbose >= 2: \
                    print("Accepted at", tt + 1)
                    break
                else:
                    L *= self.eta

            if self.max_steps == 0:
                coefx = coef - G / L
                coefx = penalty.projection(coefx, self.alpha, L)

            t = (1 + np.sqrt(1 + 4 * t_old * t_old) / 2)
            coef = coefx + (t_old - 1) / t * (coefx - coefx_old)
            df = safe_sparse_dot(X, coef.T)

            # Callback might need self.coef_.
            self.coef_ = coef
            if self.callback is not None:
                ret = self.callback(self)
                if ret is not None:
                    break

        return self
