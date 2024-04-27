import numpy as np
import itertools

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
                    vector[list(set(idx)-set(indices))] = 1
                    result.append(vector)
            M = np.array(result)
            name_list =  []
            for i in range(M.shape[0]):
                name_list.append(str(M[i,:])+'|'+str([]))
            return name_list

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
    
    postion_idx = [idx_full.index(item) for item in idx_partial]
    prob_list = {}
    for i in range(sub_binary_matrix.shape[0]):
        name = str(sub_binary_matrix[i,:])+'|'+str([])
        prob_list[name] = sub_prob_matrix[i,0]
    numerator_name = get_numerator(l =len_full, idx = postion_idx)
    denominator_name = get_denominator(l =len_full, idx= postion_idx)
    numerator = 1
    denominator = 1
    for name in numerator_name:
        numerator *= prob_list[name]
    for name in denominator_name:
            denominator *= prob_list[name]
    return np.log(numerator/denominator)

def get_name_list(l,idx):
    numerator_name = get_numerator(l = l,idx = idx)
    denominator_name = get_denominator(l = l,idx = idx)
    name_list = {}
    name_list['numerator'] = numerator_name
    name_list['denominator'] = denominator_name
    #print(name_list) 
    print(f'idx set: {idx}')
    for name, lists in name_list.items():
         print(f"{name}:")
         for ele in lists:
             print(ele)
if __name__ == '__main__':
    W = np.array([[0,1,1,-0.5],[0,0,-1,1],[0,0,0,1],[0,0,0,0]])
    d = W.shape[0]
    binary_matrix = generate_binary_matrix(d)
    p0 = 0.5
    prob_matrix = prob(binary_matrix,p0,W)
    idx_full = [3,2,1]
    idx_partial = [2,1]
    f = get_f(binary_matrix=binary_matrix, prob_matrix=prob_matrix, idx_full=idx_full, idx_partial=idx_partial)
    print(f"id_full: {idx_full} \n idx_partial: {idx_partial} \n f: {f}")
    