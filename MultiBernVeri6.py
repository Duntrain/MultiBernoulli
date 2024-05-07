import numpy as np
import itertools
import math
import pprint
import copy
import warnings

def layer_decomposition(W):
    B = (W==0)
    d = W.shape[0]
    layer_decom = []
    layer_decoms = []
    count = 0
    Omega = list(range(d))
    while bool(Omega):
        layer = list(np.where(B.all(axis=0))[0])
        layer_decoms.append(layer)
        if count!=0:
            layer = [x for x in layer if (x not in layer_decoms[count - 1])]
        for i in layer:
            B[i,:] = True
        layer_decom.append(layer)
        Omega = [x for x in Omega if (x not in layer)]
        count += 1
    return layer_decom

def is_no_cross_term_dict_consistent_with_layers(layer_decom,no_cross_term_dict):
    for topo in no_cross_term_dict.keys():
        topo = eval(topo)
        if not is_consistent_topo_with_layers(topo,layer_decom):
            return False
    return True
def is_consistent_topo_with_layers(topo, layer_decom):
    # Create a dictionary to map each node to its index in the topological sort
    index_in_topo = {node: i for i, node in enumerate(topo)}
    
    # Check if all nodes in each layer appear before any node in the next layer
    last_index_in_previous_layers = -1  # No nodes before the first layer
    for layer in layer_decom:
        # Find the minimum and maximum indices of the current layer nodes in the topological sort
        min_index = min(index_in_topo[node] for node in layer)
        max_index = max(index_in_topo[node] for node in layer)
        
        # Ensure all nodes in this layer are after all nodes in previous layers
        if min_index <= last_index_in_previous_layers:
            return False
        
        # Update last index for the next layer comparison
        last_index_in_previous_layers = max_index

    return True
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
    ratio = 1
    for name1, name2 in zip(numerator_name,denominator_name):
        ratio *= prob_list[name1]/prob_list[name2]
    return np.log(ratio)

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


def get_all_f(prob_matrix):
    d = int(math.log2(prob_matrix.shape[0]))
    binary_matrix = generate_binary_matrix(d)
    all_f = {}
    for i in range(3,d+1):
        for idx_full in itertools.combinations(list(range(0,d)),i):
            idx_full = list(idx_full)
            for l in range(3,i+1):
                for idx_partial in itertools.combinations(idx_full,l):
                    idx_partial = list(idx_partial)
                    f = get_f(binary_matrix=binary_matrix, 
                                prob_matrix=prob_matrix, 
                                idx_full=idx_full, 
                                idx_partial=idx_partial) 
                    if not np.isclose(f,0,atol=1e-13):
                        all_f[str(idx_full)+'|'+str(idx_partial)] = f
                    else:
                        all_f[str(idx_full)+'|'+str(idx_partial)] = 0
    return all_f






def check_cross_term_topo_fast(topo, prob_matrix, all_f):
    # this time, we return whether the topo has cross term or not, and we don't calculate the f
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
                if all_f is not None:
                    f = all_f[str(sorted(idx_full))+'|'+str(sorted(idx_partial))]
                else:
                    f = get_f(binary_matrix=binary_matrix, 
                                prob_matrix=prob_matrix, 
                                idx_full=idx_full, 
                                idx_partial=idx_partial)
                if not np.isclose(f,0,atol=1e-13):
                    cross_term_dict_details[str(idx_full)+'|'+str(idx_partial)] = f
                    return cross_term_dict_details
    return cross_term_dict_details

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
                if not np.isclose(f,0,atol=1e-13):
                    cross_term_dict_details[str(idx_full)+'|'+str(idx_partial)] = f
    
    return cross_term_dict_details

def check_cross_term_fast(prob_matrix,verbose = False,use_all_f = True):
    # this fast version of check_cross_term: once it find topo that is not list(range(0,d)) and has no cross term, it will stop searching
    if use_all_f:
        all_f = get_all_f(prob_matrix)
    else:
        all_f = None
    vprint = print if verbose else lambda *a, **k: None
    d = int(math.log2(prob_matrix.shape[0]))
    no_cross_term = True
    cross_term_dict = {}
    no_cross_term_dict = {}
    for topo in itertools.permutations(list(range(0,d))):
        topo = list(topo)
        if set(topo[:3]) == set([0,1,2]) and not topo[:3] == [0,1,2]:
            continue
        else:
            cross_term_dict_details = check_cross_term_topo_fast(topo = topo, prob_matrix=prob_matrix,all_f=all_f)
            if len(cross_term_dict_details)>0:
                cross_term_dict[str(topo)] = 'Cross term found'
                no_cross_term = False
                vprint(f"Topo has cross-term: {topo}")
                vprint(cross_term_dict_details)
                vprint('-----------------------------------')
            else:
                if topo!= list(range(0,d)):
                    if no_cross_term_dict.get(str(list(range(0,d)))) is None:
                        no_cross_term_dict[str(list(range(0,d)))] = 'No cross term found'
                    no_cross_term_dict[str(topo)] = 'No cross term found'
                    return cross_term_dict,no_cross_term_dict
                else:
                    no_cross_term_dict[str(topo)] = 'No cross term found'
    # if no_cross_term:
    #     vprint("No cross term found for all permuation")
    return cross_term_dict,no_cross_term_dict


def check_cross_term(prob_matrix,verbose = False,use_all_f = True):
    if use_all_f:
        all_f = get_all_f(prob_matrix)
    else:
        all_f = None
    vprint = print if verbose else lambda *a, **k: None
    d = int(math.log2(prob_matrix.shape[0]))
    no_cross_term = True
    cross_term_dict = {}
    no_cross_term_dict = {}
    for topo in itertools.permutations(list(range(0,d))):
        topo = list(topo)
        if set(topo[:3]) == set([0,1,2]) and not topo[:3] == [0,1,2]:
            continue
        else:
            cross_term_dict_details = check_cross_term_topo_fast(topo = topo, prob_matrix=prob_matrix,all_f=all_f)
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

def remove_edges(matrix, num_edges_to_remove = -1,indices = None):
    """ Generate all matrices after removing combinations of edges. """
    d = matrix.shape[0]
    # if indices is None:
    #     indices = upper_triangle_indices(d)
    all_matrices = []
    if num_edges_to_remove == -1:
        print("Removing all possible edges")
        # Generate all combinations of edges to be removed
        if indices is None:
            indices = upper_triangle_indices(d)
            for num_edges_to_remove in range(len(indices), 0, -1):
                for combination in itertools.combinations(indices, num_edges_to_remove):
                    # Create a copy of the matrix to modify
                    modified_matrix = copy.deepcopy(matrix)
                    # Remove edges by setting the corresponding indices to zero
                    for index in combination:
                        modified_matrix[index] = 0
                    all_matrices.append(modified_matrix)
        else:
            for num_edges_to_remove in range(len(indices), 0, -1):
                for combination in itertools.combinations(indices, num_edges_to_remove):
                    # Create a copy of the matrix to modify
                    modified_matrix = copy.deepcopy(matrix)
                    # Remove edges by setting the corresponding indices to zero
                    for index in combination:
                        modified_matrix[index] = 0
                    all_matrices.append(modified_matrix)

    else:
        print(f"Removing {num_edges_to_remove} edges")
        for combination in itertools.combinations(indices, num_edges_to_remove):
                # Create a copy of the matrix to modify
                modified_matrix = copy.deepcopy(matrix)
                # Remove edges by setting the corresponding indices to zero
                for index in combination:
                    modified_matrix[index] = 0
                all_matrices.append(modified_matrix)
    return all_matrices

def is_supmatrix_list(A,matrices_list):
    # check whether A is subgraph of B
    for B in matrices_list:
        if np.all((B!=0)<=(A!=0)):
            return True
    return False

def test1(d = 4):
    np.random.seed(0)
    W = generate_random_W(d)
    # W = np.array([[0, 0, 0, 0],
    #               [0, 0, 0,-1],
    #               [0, 0, 0, 1],
    #               [0, 0, 0, 0]])
    binary_matrix = generate_binary_matrix(d)
    p0 = np.random.uniform(low = 0.3, high = 0.7, size = (d))
    print(f"p0: {p0}")
    resulting_matrices = remove_edges(W, num_edges_to_remove = -1)
    success_matrice = []
    for matrix in resulting_matrices:
        prob_matrix = prob(binary_matrix,p0,matrix)
        _, no_cross_term_dict = check_cross_term(prob_matrix)
        layer_decom = layer_decomposition(matrix)
        consistent = is_no_cross_term_dict_consistent_with_layers(layer_decom,no_cross_term_dict)
        if consistent:
            success_matrice.append(matrix)
            print("Matrix is consistent with no cross term dict")
            print(f"Matrix: \n{matrix}")

    print(f"Number of successful matrices: {len(success_matrice)}")
    pprint.pprint(success_matrice)


def test2():
    np.random.seed(0)
    W = np.array([[0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 1],
                  [0, 0, 0, 0, 0]])
    d = W.shape[0]
    binary_matrix = generate_binary_matrix(d)
    p0 = np.random.uniform(low = 0.3, high = 0.7, size = (d))
    print(f"p0: {p0}")
    prob_matrix = prob(binary_matrix,p0,W)
    _, no_cross_term_dict = check_cross_term(prob_matrix)
    layer_decom = layer_decomposition(W)
    consistent = is_no_cross_term_dict_consistent_with_layers(layer_decom,no_cross_term_dict)
    pprint.pprint(f"Consistent: {consistent}")
    pprint.pprint(f"Layer decomposition: {layer_decom}")
    # print(f"No cross term dict: {no_cross_term_dict}")
    print("No cross term dict:")
    pprint.pprint(f"{no_cross_term_dict}")
if __name__ == '__main__':
    test1(d = 5)