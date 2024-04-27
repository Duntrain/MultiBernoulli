import numpy as np
import itertools
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def prob(x,p0,W):
    # x is a 1 by p vector
    # W is triupper matrix of size p by p
    xw = x@ W
    p = sigmoid(xw)
    p[:,0] = p0
    return np.prod(p**x*(1-p)**(1-x),axis=1,keepdims=True)
def generate_binary_matrix(d):
    # Generate all combinations of 0s and 1s for d columns
    combinations = itertools.product([0, 1], repeat=d)
    
    # Convert the combinations to a NumPy array
    matrix = np.array(list(combinations))
    return matrix

def get_f(binary_matrix, prob_matrix, idx_full, idx_partial):
    # idx_full is the full set of indices
    # idx_partial is the partial set of indices
    # d is the total number of indices
    len_full = len(idx_full)
    len_partial = len(idx_partial)
    sub_binary_matrix = generate_binary_matrix(len_full)
    sub_prob_matrix = np.zeros((2**len_full,1))
    
    for i in range(2**len_full):
        pos_true = np.argwhere((binary_matrix[:,idx_full] == sub_binary_matrix[i,:]).all(axis=1))
        sub_prob_matrix[i,0] = np.sum(prob_matrix[pos_true,0])
    
if __name__ == '__main__':
    W = np.array([[0,1,0],[0,0,1],[0,0,0]])
    d = W.shape[0]
    binary_matrix = generate_binary_matrix(d)
    p0 = 0.5
    print(binary_matrix)
    prob_matrix = prob(binary_matrix,p0,W)
    print(prob_matrix)
    get_f(binary_matrix=binary_matrix, prob_matrix=prob_matrix, idx_full=[0,1], idx_partial=[0])
    
    