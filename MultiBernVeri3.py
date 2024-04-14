# this file is used to deal with special case when we only have 3 variables and 2 edges as chain graph.

import numpy as np
from sklearn.linear_model import LogisticRegression
from itertools import combinations
import itertools
from sklearn.metrics import log_loss
import math
import pprint
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
                                solver='newton-cholesky',max_iter=5000)  
         
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

def get_conditional_prob(X, idx_a, idx_b):
    # get conditional probability of P(X_a|X_b)
    n, d = X.shape
    len_a, len_b = len(idx_a), len(idx_b)
    len_ab = len_a + len_b
    prob_dict = {}
    def generate_matrix(l):
        # Generate all possible combinations of 0s and 1s for length l
        combinations = list(itertools.product([0, 1], repeat=l))
        
        # Convert list of tuples into a numpy array to form the matrix
        matrix = np.array(combinations)
        return matrix
    M = generate_matrix(len_ab)
    for i in range(2**len_ab):
        name = str(M[i,:len_a])+'|'+str(M[i,len_a:])
        X_cond = np.hstack((X[:,idx_a],X[:,idx_b]))
        prob_dict[name] = np.sum(np.all(X_cond == M[i,:], axis=1))/np.sum(np.all(X_cond[:,len_a:] == M[i,len_a:], axis=1))
    return prob_dict

def get_f(prob_list,idx):
        l = int(math.log2(len(prob_list)))

        def get_numerator(l, idx):
            # Resultant list of valid vectors
            result = []
            len_idx = len(idx)
            for r in range(0,len_idx+1,2):
                for indices in itertools.combinations(idx, r):
                    vector = np.zeros(l, dtype=int)
                    vector[list(set(idx)-set(indices))] = 1
                    result.append(vector)
            M = np.array(result)
            name_list =  []
            for i in range(M.shape[0]):
                name_list.append(str(M[i,:])+'|'+str([]))
            return name_list
        def get_denominator(l, idx):
            # Resultant list of valid vectors
            result = []
            len_idx = len(idx)
            for r in range(1,len_idx+1,2):
                for indices in itertools.combinations(idx, r):
                    vector = np.zeros(l, dtype=int)
                    vector[list(indices)] = 1
                    result.append(vector)
            M = np.array(result)
            name_list =  []
            for i in range(M.shape[0]):
                name_list.append(str(M[i,:])+'|'+str([]))
            return name_list
        
        numerator_name = get_numerator(l=  l,idx = idx)
        denominator_name = get_denominator(l = l,idx = idx)
        numerator = 1
        denominator = 1
        for name in numerator_name:
            numerator *= prob_list[name]


        for name in denominator_name:
            denominator *= prob_list[name]

        return np.log(numerator/denominator)

if __name__ == '__main__':
    import utils
    from Topo_utils import create_Z, find_topo
    import warnings
    warnings.filterwarnings("ignore")
    rd_int = np.random.randint(10000, size=1)[0]
    rd_int  = 123
    print(rd_int)
    utils.set_random_seed(rd_int)
    n, d, s0 = 2000000, 3, 2
    graph_type, sem_type = 'ER', 'logistic'

    # B_true = utils.simulate_dag(d, s0, graph_type)
    # W_true = np.array([[0, -2, 2], [0, 0, 0], [0, 0, 0]])
    # topo = [0,1,2]
    # j = 2
    

    # # W_true = utils.simulate_parameter(B_true, w_ranges=((-30.0, -10), (10, 30)))
    # X = utils.simulate_linear_sem(W_true, n, sem_type)
    # score, dict_coef, v = regress(X = X ,topo = topo ,j = j)
    # print(f"score: {score}")
    # print("dict_coef: ")
    # pprint.pprint(dict_coef)
    # print(f"v: {v}")

    # prob_list = get_conditional_prob(X, [0,1,2], [])
    # print("prob_list: ")
    # pprint.pprint(prob_list)# print all elements in prob_list, how to do it?

    # print(f"f[0,1,2]->[0,1]: {get_f(prob_list, [0,1,2])}")
    # print(f"f[0,2]->[0]: {get_f(prob_list, [0,2])}")
    # print(f"f[1,2]->[1]: {get_f(prob_list, [1,2])}")
    



    W_true = np.array([[0, -2, 0, 2], [0, 0, 1, 0], [0, 0, 0, 0.5],[0, 0, 0, 0]])
    topo = [0,1,2,3]
    j = 3

    X = utils.simulate_linear_sem(W_true, n, sem_type)
    score, dict_coef, v = regress(X = X ,topo = topo ,j = j)
    print(f"score: {score}")
    print("dict_coef: ")
    pprint.pprint(dict_coef)
    print(f"v: {v}")

    prob_list = get_conditional_prob(X, [0,1,2,3], [])
    print("prob_list: ")
    pprint.pprint(prob_list)# print all elements in prob_list, how to do it?

    print(f"f[0,1,2,3]->[0,1,2]: {get_f(prob_list, [0,1,2,3])}")
    print(f"f[0,3]->[0]: {get_f(prob_list, [0,3])}")
    print(f"f[1,3]->[1]: {get_f(prob_list, [1,3])}")
    print(f"f[2,3]->[2]: {get_f(prob_list, [2,3])}")