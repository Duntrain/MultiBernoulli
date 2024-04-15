import numpy as np
import itertools

# def generate_vectors(l, idx):
#     # Resultant list of valid vectors
#     result = []
#     len_idx = len(idx)
#     for r in range(0,len_idx+1,2):
#         for indices in itertools.combinations(idx, r):
#             vector = np.zeros(l, dtype=int)
#             vector[list(set(idx)-set(indices))] = 1
#             result.append(vector)
#     return np.array(result)

# # Example usage
# l = 3
# idx = [0, 1,2]  # Zero-based indexing
# vectors = generate_vectors(l, idx)
# print("Generated vectors with an even number of zeros in idx:")
# print(vectors)



#

def tab_(p1,w1,w2,w3,x):
    x1,x2,x3 = x
    prob = 1
    prob = prob* p1**x1*(1-p1)**(1-x1)
    prob = prob*(1/(1+np.exp(-w1*x1)))**x2*(1 -1/(1+np.exp(-w1*x1)))**(1-x2)
    prob = prob* (1/(1+np.exp(-w2*x1-w3*x2)))**x3*(1 -1/(1+np.exp(-w2*x1-w3*x2)))**(1-x3)
    return prob

def get_prob_list(p1,w1,w2,w3):
    prob_list = {}
    for x in itertools.product([0, 1], repeat=3):
        prob_list[str(x)] = tab_(p1,w1,w2,w3,x)
    return prob_list

# Example usage
p1, w1, w2, w3 = 0.5, -1, 0, 1
prob_list = get_prob_list(p1, w1, w2, w3)
print("Generated probability list:")
print(prob_list)
# calculate E[(X_1-E[X_1])(X_2-E[X_2])(X_3-E[X_3])]
def get_cov(p1,w1,w2,w3):
    tab = lambda x: tab_(p1,w1,w2,w3,x)
    E_X1 = p1
    E_X2 = tab([0,1,0])+tab([1,1,0])+tab([0,1,1])+tab([1,1,1])
    E_X3 = tab([0,0,1])+tab([1,1,1])+tab([0,1,1])+tab([1,0,1])
    E_X1X2 = tab([1,1,0])+tab([1,1,1])
    E_X1X3 = tab([1,0,1])+tab([1,1,1])
    E_X2X3 = tab([0,1,1])+tab([1,1,1])
    E_X1X2X3 = tab([1,1,1])

    cov = E_X1X2X3 - E_X1X2*E_X3 - E_X1X3*E_X2 - E_X2X3*E_X1 + 2*E_X1*E_X3*E_X2
    f_123 = (tab([1,1,1])*tab([0,0,1])*tab([1,0,0])*tab([0,1,0]))/(tab([1,1,0])*tab([1,0,1])*tab([0,1,1])*tab([0,0,0]))
    return cov,np.log(f_123)
cov,f_123 = get_cov(p1,w1,w2,w3)
print("Covariance: ", cov)
print("f_123: ", f_123)