# %%
from Topo_utils import threshold_W,create_Z,find_idx_set,create_new_topo,create_new_topo_greedy
import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression
import numpy.linalg as la

import scipy.linalg as slin
from copy import copy
class TOPO_linear:
    def __init__(self,score,regress):
        super().__init__()
        self.score = score
        self.regress = regress

    def _init_W_slice(self,idx_y,idx_x):
        y = self.X[:, idx_y]
        x = self.X[:, idx_x]
        w = self.regress(X = x, y = y)
        return w

    def _init_W(self,Z):
        W = np.zeros((self.d, self.d))
        for j in range(self.d):
            if (~Z[:, j]).any():
                W[~Z[:, j], j] = self.regress(X = X[:, ~Z[:, j]], y =X[:, j])
            else:
                W[:, j] = 0
        return W

    def _h(self, W):
        """Evaluate value and gradient of acyclicity constraint.
        Option 1: h(W) = Tr(I+|W|/d)^d-d
        """

        """
        h(W) = -log det(sI-W*W) + d log (s)
        nabla h(W) = 2 (sI-W*W)^{-T}*W
        """
        I = np.eye(self.d)
        s = 1
        M = s * I - np.abs(W)
        h = - np.linalg.slogdet(M)[1] + self.d * np.log(s)
        G_h = slin.inv(M).T

        return h, G_h

    def _update_topo_linear(self, W, topo, idx, opt = 1):

        topo0 = copy(topo)
        W0 = np.zeros_like(W)
        i,j = idx
        i_pos,j_pos = topo.index(i), topo.index(j)

        W0[:, topo[:j_pos]] = W[:,topo[:j_pos]]
        W0[:, topo[(i_pos+1):]] = W[:, topo[(i_pos+1):]]
        topo0 = create_new_topo(topo = topo0, idx = idx, opt = opt)
        for k in range(j_pos,i_pos+1):
            if len(topo0[:k])!=0:
                W0[topo0[:k],topo0[k]] = self._init_W_slice(idx_y=topo0[k],idx_x=topo0[:k])
            else:
                W0[:,topo0[k]] = 0
        return W0,topo0

    def _update(self,loss,idx_set:list,W,topo,opt = 1):
        succ = False
        idx_set = list(idx_set)
        idx_len = len(idx_set)
        loss_collections = np.zeros(idx_len)
        for i in range(idx_len):
            W_c, topo_c = self._update_topo_linear(W=W, topo=topo, idx=idx_set[i], opt = opt)
            loss_c, _ = self.score(X=self.X, W=W_c)
            loss_collections[i] = loss_c

        if np.any(loss > np.min(loss_collections)):
            topo = create_new_topo_greedy(self.topo, loss_collections, idx_set, loss, opt = opt)
            succ = True
        return topo, succ

    def fit(self, X, topo : list,no_large_search,size_small,size_large):
        self.n, self.d = X.shape
        self.X = X
        iter_count = 0 # iteration count
        large_space_used = 0
        if not isinstance(topo, list):
            raise TypeError
        else:
            self.topo = topo

        Z = create_Z(self.topo)
        self.Z = Z
        self.W = self._init_W(self.Z)
        loss, G_loss = self.score(X = self.X, W = self.W)
        h, G_h = self._h(W= self.W)
        idx_set_small, idx_set_large  = find_idx_set(G_h=G_h, G_loss= G_loss, Z=self.Z, size_small=size_small, size_large=size_large)
        idx_set = list(idx_set_small)
        while bool(idx_set):
            topo_possible, succ = self._update(loss = loss, idx_set =idx_set, W = self.W, topo = self.topo, opt = 1)
            if succ:
                self.topo = topo_possible
                print("in")
            else:
                print("out")
                if large_space_used < no_large_search:
                    idx_set = idx_set_large.difference(idx_set_small)
                    idx_set = list(idx_set)

                    topo_possible, succ = self._update(loss=loss, idx_set=idx_set, W=self.W, topo=self.topo, opt=1)
                    if succ:
                        self.topo = topo_possible
                        large_space_used += 1
                    else:
                        topo_possible, succ = self._update(loss=loss, idx_set=idx_set, W=self.W, topo=self.topo, opt=2)
                        if succ:
                            self.topo = topo_possible
                            large_space_used += 1
                        else:
                            topo_possible, succ = self._update(loss=loss, idx_set=idx_set, W=self.W, topo=self.topo,
                                                               opt=3)
                            if succ:
                                self.topo = topo_possible
                                large_space_used += 1
                            else:
                                print("We cannot find better loss")
                                break
                else:
                    print("We reach the number of chances to search large space, it is {}".format(
                        no_large_search))
                    break

            self.Z = create_Z(self.topo)
            self.W = self._init_W(self.Z)
            loss, G_loss = self.score(X=self.X, W=self.W)
            h, G_h = self._h(W=self.W)
            idx_set_small, idx_set_large = find_idx_set(G_h=G_h, G_loss=G_loss, Z=self.Z, size_small=size_small,
                                                        size_large=size_large)
            idx_set = list(idx_set_small)

            iter_count += 1
            print(iter_count)

        return self.W, self.topo, Z, loss

if __name__ == '__main__':
    import utils
    from timeit import default_timer as timer

    rd_int = int(np.random.randint(10000, size=1))
    rd_int = 12
    utils.set_random_seed(rd_int)

    n, d, s0 = 1000, 10, 20
    graph_type, sem_type = 'ER', 'gauss'

    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true)
    X = utils.simulate_linear_sem(W_true, n, sem_type)

    ## Linear Model
    def regress(X,y):
        reg = LinearRegression(fit_intercept=False)
        reg.fit(X = X, y = y)
        return reg.coef_
    def score(X,W):
        M = X @ W
        R = X - M
        loss = 0.5 / X.shape[0] *(R **2 ).sum()
        G_loss = - 1.0 / X.shape[0] * X.T @ R

        return loss, G_loss
    '''
    ## Logistic Model
    def regress(X,y,C = 0.1):
        reg = LogisticRegression(multi_class='ovr', fit_intercept=False, penalty=penalty, C=C,
                                 solver='liblinear')                         
        reg.fit(X = X, y = y)
        return reg.coef_
    
    def score(X,W,C = 0.1):
        lambda1 = 1/C
        M = X @ W
        loss = 1.0 / X.shape[0] * (np.logaddexp(0, M) - X * M).sum() + lambda1 * (np.abs(W)).sum()
        G_loss1 = 1.0 / X.shape[0] * X.T @ (sigmoid(M) - X)
        G_loss = G_loss1 + gradient_l1(W, G_loss1, lambda1)
        return loss, G_loss
    
    '''


    model = TOPO_linear(regress=regress, score = score)
    topo_init = list(np.random.permutation(range(d)))
    start = timer()
    W_est,_,_,_ = model.fit(X = X,topo = topo_init, no_large_search=10,size_small=44, size_large=45)
    end = timer()
    acc = utils.count_accuracy(B_true, threshold_W(W = W_est) != 0)
    print(acc)
    print(f'time: {end - start:.4f}s')





