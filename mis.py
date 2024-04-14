import numpy as np
import itertools

def generate_vectors(l, idx):
    # Resultant list of valid vectors
    result = []
    len_idx = len(idx)
    for r in range(0,len_idx+1,2):
        for indices in itertools.combinations(idx, r):
            vector = np.zeros(l, dtype=int)
            vector[list(set(idx)-set(indices))] = 1
            result.append(vector)
    # # Iterate over all possible numbers of 1's that would result in an even number of zeros
    # for r in range(1, n+1, 2):  # Only odd numbers of 1's (even number of 0's)
    #     for indices in itertools.combinations(idx, r):
    #         # Create a zero vector of length l
    #         vector = np.zeros(l, dtype=int)
    #         # Set elements at selected indices to 1
    #         vector[list(indices)] = 1
    #         result.append(vector)
    
    return np.array(result)

# Example usage
l = 3
idx = [0, 1,2]  # Zero-based indexing
vectors = generate_vectors(l, idx)
print("Generated vectors with an even number of zeros in idx:")
print(vectors)
