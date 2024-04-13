# this file is used to deal with special case when we only have 3 variables and 2 edges as chain graph.

import numpy as np
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import itertools
from sklearn.metrics import log_loss

def expand_matrix(matrix):
    n, p = matrix.shape
    # Initialize the list of new columns with the original matrix
    new_columns = [matrix[:, [i]] for i in range(p)]
    
    # Generate and multiply combinations of columns
    for r in range(2, p+1):  # Starting from 2 as we already have individual columns
        for indices in combinations(range(p), r):
            # Multiply elements across the specified columns and create a new column
            product_column = np.prod([matrix[:, i] for i in indices], axis=0).reshape(n, 1)
            new_columns.append(product_column)
    
    # Concatenate all columns to form the expanded matrix
    expanded_matrix = np.hstack(new_columns)
    return expanded_matrix


def regress(X,topo,j):
    d = len(topo)
    if j == 0:
        X_expd = np.ones((n, 1))
    else:
        X_expd = expand_matrix(X[:, topo[:j]])
        X_expd = np.hstack([np.ones((n, 1)), X_expd])
    reg = LogisticRegression(multi_class='ovr', fit_intercept=False, penalty='none',
                                solver='lbfgs',max_iter=5000)       
    reg.fit(X = X_expd, y = X[:, topo[j]])
    prob = reg.predict_proba(X_expd)[:,1]
    score = -log_loss(X[:, topo[j]],prob)
    v = np.zeros(d)
    dict_coef = {}
    if j>0:
        idx_col = [[topo[i]] for i in range(j)]
        for r in range(2, j+1):
            for indices in combinations(topo[:j], r):
                idx_col.append(list(indices))
    for i in range(len(reg.coef_[0])):
        if i == 0:
            dict_coef['const'] = reg.coef_[0][i]
        else:
            dict_coef[str(idx_col[(i-1)])] = reg.coef_[0][i]
    for pos_i, sublist in enumerate(idx_col):
            for ele in sublist:
                v[ele]+=abs(reg.coef_[0][pos_i+1])
    # v = np.zeros(d)
    # if j>0:
    #     idx_col = [[topo[i]] for i in range(j)]
    #     for r in range(2, j+1):
    #         for indices in combinations(topo[:j], r):
    #             idx_col.append(list(indices))
    #     for pos_i, sublist in enumerate(idx_col):
    #         for ele in sublist:
    #             v[ele]+=abs(reg.coef_[0][pos_i+1])
    return score,dict_coef,v

if __name__ == '__main__':
    import utils
    from Topo_utils import create_Z, find_topo
    import warnings
    warnings.filterwarnings("ignore")
    rd_int = np.random.randint(10000, size=1)[0]
    rd_int  = 123
    print(rd_int)
    utils.set_random_seed(rd_int)
    n, d, s0 = 1000000, 3, 2
    graph_type, sem_type = 'ER', 'logistic'

    # B_true = utils.simulate_dag(d, s0, graph_type)
    # W_true = np.array([[0, 10, 0], [0, 0, 10], [0, 0, 0]])
    # topo = [0,1,2]
    # j = 2
    W_true = np.array([[0, 10, 0, 10], [0, 0, 10,0], [0, 0, 0,10],[0,0,0,0]])
    topo = [0,1,2,3]
    j = 3

    # W_true = utils.simulate_parameter(B_true, w_ranges=((-30.0, -10), (10, 30)))
    X = utils.simulate_linear_sem(W_true, n, sem_type)
    score, dict_coef, v = regress(X = X ,topo = topo ,j = j)
    print(f"score: {score}, dict_coef: {dict_coef}")
    print(f"v: {v}")


