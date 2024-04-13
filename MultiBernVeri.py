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

def regress(X,y):
        n = y.shape[0]
        if X.size ==0:
            X_expd = np.ones((n, 1))
        else:
            X_expd = expand_matrix(X)
            X_expd = np.hstack([np.ones((n, 1)), X_expd])
        reg = LogisticRegression(multi_class='ovr', fit_intercept=False, penalty='none',
                                 solver='lbfgs',max_iter=2000)                         
        reg.fit(X = X_expd, y = y)
        # print(f"reg.coef_ {reg.coef_}")
        prob = reg.predict_proba(X_expd)[:,1]
        score = -log_loss(y,prob)
        return score


if __name__ == '__main__':
    import utils
    from Topo_utils import create_Z, find_topo
    import warnings
    warnings.filterwarnings("ignore")
    rd_int = np.random.randint(10000, size=1)[0]
    print(rd_int)
    utils.set_random_seed(rd_int)
    n, d, s0 = 1000000, 5, 5
    graph_type, sem_type = 'ER', 'logistic'
    B_true = utils.simulate_dag(d, s0, graph_type)
    W_true = utils.simulate_parameter(B_true, w_ranges=((-30.0, -10), (10, 30)))
    X = utils.simulate_linear_sem(W_true, n, sem_type)

    def _get_score(Z,X):
        score = 0
        W = np.zeros_like(Z)
        for j in range(d):
            if (~Z[:, j]).any():
                score+=  regress(X=X[:, ~Z[:, j]], y=X[:, j])
            else:
                score+= regress(X = np.empty((0, 0)),y = X[:,j])
        return score
    
    print(f"True W \n{W_true}  \n True B \n {B_true} \n True topo \n {find_topo(W_true)} \n")
    
    for perm in list(itertools.permutations(list(range(d)))):
        topo_init = list(perm)
        # topo_init = find_topo(W_true)
        Z = create_Z(topo_init)
        # print(f"current topo \n {topo_init} \n ")
        score  = _get_score(Z, X)
        print(f"current topo: {topo_init}, current score: {score}")

