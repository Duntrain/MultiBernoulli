import numpy as np
import itertools
import math
import pprint
import copy
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
    zero_columns = np.all(W == 0, axis=0)
    p = sigmoid(xw)
    p[:,zero_columns] = p0[zero_columns]
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

def check_cross_term_topo(topo, prob_matrix):
    # topo = [0,1,2,3,4,5,6,7]
    cross_term_dict_details = {}
    d = len(topo)
    assert d>=3, "The number of nodes must be greater than or equal to 3"
    binary_matrix = generate_binary_matrix(d)
    for i in range(3,d+1):
        idx_full = topo[:i]
        idx_med = topo[:(i-1)]
        for l in range(2,i):
            for item in itertools.combinations(idx_med,l):
                idx_partial = list(item)+[idx_full[-1]]
                f = get_f(binary_matrix=binary_matrix, 
                            prob_matrix=prob_matrix, 
                            idx_full=idx_full, 
                            idx_partial=idx_partial)
                if not np.isclose(f,0,atol=1e-9):
                        cross_term_dict_details[str(idx_full)+'|'+str(idx_partial)] = f
                        
                        

                # print(f"id_full: {idx_full} \nidx_partial: {idx_partial} \n f: {f}")
                # # get_name_list(l = d, idx = idx_full)
                # # get_name_list(l = d, idx = idx_partial)
                # print('-----------------------------------')
    
    return cross_term_dict_details

def check_cross_term(prob_matrix,verbose = False):
    vprint = print if verbose else lambda *a, **k: None
    d = int(math.log2(prob_matrix.shape[0]))
    no_cross_term = True
    cross_term_dict = {}
    no_cross_term_dict = {}
    for topo in itertools.permutations(list(range(d))):
        topo = list(topo)
        cross_term_dict_details = check_cross_term_topo(topo = topo, prob_matrix=prob_matrix)
        if len(cross_term_dict_details)>0:
            cross_term_dict[str(topo)] = 'Cross term found'
            no_cross_term = False
            vprint(f"Topo has cross-term: {topo}")
            vprint(cross_term_dict_details)
            vprint('-----------------------------------')
        else:
            no_cross_term_dict[str(topo)] = 'No cross term found'
    if no_cross_term:
        vprint("No cross term found for all permuation")

    return cross_term_dict,no_cross_term_dict
def generate_random_W(d):
    W = np.zeros((d, d))
    for i in range(d):
        for j in range(i+1,d):
            # Choose a range randomly
            if np.random.rand() > 0.5:
                # Sample from [0.5, 2]
                W[i, j] = np.random.uniform(0.5, 2)
            else:
                # Sample from [-2, -0.5]
                W[i, j] = np.random.uniform(-2, -0.5)
    
    return W

def upper_triangle_indices(d):
    """ Generate indices of the upper triangle of a dxd matrix (excluding the diagonal). """
    return [(i, j) for i in range(d) for j in range(i + 1, d)]

def remove_edges(matrix):
    """ Generate all matrices after removing combinations of edges. """
    d = matrix.shape[0]
    indices = upper_triangle_indices(d)
    all_matrices = []

    # Generate all combinations of edges to be removed
    for num_edges_to_remove in range(1, len(indices) + 1):
        for combination in itertools.combinations(indices, num_edges_to_remove):
            # Create a copy of the matrix to modify
            modified_matrix = copy.deepcopy(matrix)
            # Remove edges by setting the corresponding indices to zero
            for index in combination:
                modified_matrix[index] = 0
            all_matrices.append(modified_matrix)
    
    return all_matrices
if __name__ == '__main__':
    np.random.seed(1)
    # W = np.array([[0,1,0,0],[0,0,0,1],[0,0,0,-1],[0,0,0,0]]) # interesting case with v structure
    # W = np.array([[0,1,-0.5,0],[0,0,0,1],[0,0,0,-1],[0,0,0,0]])
    
    # W is chain with five node
    # W = np.array([[0,1,0,0.5,0],[0,0,-1,0,0],[0,0,0,0.5,0],[0,0,0,0,1],[0,0,0,0,0]])

    W = generate_random_W(d = 4)
    
    # W = np.array([[0,0,1],[0,0,-1],[0,0,0]])
    d = W.shape[0]
    binary_matrix = generate_binary_matrix(d)
    p0 = np.random.uniform(low = 0.3, high = 0.7, size = (d))
    print(f"p0: {p0}")
    prob_matrix = prob(binary_matrix,p0,W)
    # print(f"prob_matrix: {prob_matrix}")
    cross_term_dict,no_cross_term_dict = check_cross_term(prob_matrix,verbose=True)
    pprint.pprint(cross_term_dict)
    pprint.pprint(no_cross_term_dict)
    print(f"There are {len(cross_term_dict)} topologies with cross term")
    print(f"There are {len(no_cross_term_dict)} topologies without cross term")
    print('-----------------------------------')
    # for i in range(d):
    #     for j in range(i+1,d):
    #         W_copy = W.copy()
    #         W_copy[i,j] = 0
    #         print(f"remove edge {i} -> {j}")
    #         prob_matrix = prob(binary_matrix,p0,W_copy)
    #         cross_term_dict,no_cross_term_dict = check_cross_term(prob_matrix,verbose=False)
    #         print(f"There are {len(cross_term_dict)} topologies with cross term")
    #         print(f"There are {len(no_cross_term_dict)} topologies without cross term")
    resulting_matrices = remove_edges(W)
    for matrix in resulting_matrices:
        prob_matrix = prob(binary_matrix,p0,matrix)
        cross_term_dict,no_cross_term_dict = check_cross_term(prob_matrix,verbose=False)
        # print(f"There are {len(cross_term_dict)} topologies with cross term")
        # print(f"There are {len(no_cross_term_dict)} topologies without cross term")
        if len(no_cross_term_dict) == 6:
            print('-----------------------------------')
            print("Matrix would be unique:")
            print(matrix)
         





    # check_cross_term(topo = [1,0,2], prob_matrix=prob_matrix)
    # idx_full = [0,1,2]
    # idx_partial = [0,1,2]
    # f = get_f(binary_matrix=binary_matrix, prob_matrix=prob_matrix, idx_full=idx_full, idx_partial=idx_partial)
    # print(f"id_full: {idx_full} \nidx_partial: {idx_partial} \n f: {f}")
    # idx_full1 = [0,1,2,3]
    # idx_partial1 = [0,1,3]

    # idx_full2 = [0,1,3]
    # idx_partial2 = [0,1,3]
    # f_1 = get_f(binary_matrix=binary_matrix, 
    #                    prob_matrix=prob_matrix, 
    #                    idx_full=idx_full1, 
    #                    idx_partial=idx_partial1)
    # f_2 = get_f(binary_matrix=binary_matrix, 
    #                    prob_matrix=prob_matrix, 
    #                    idx_full=idx_full2, 
    #                    idx_partial=idx_partial2)

    # difference = f_1 - f_2
    # print('Difference:',difference)

    