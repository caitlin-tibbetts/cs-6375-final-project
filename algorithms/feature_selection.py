import numpy as np

def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """
    ret = dict()
    if len(x.shape) > 0:
        for i in range(x.shape[0]):
            if x[i] in ret:
                ret[x[i]].append(i)
            else:
                ret[x[i]] = [i]
    return ret


def entropy(y):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """
    z = partition(y)
    H = 0
    for key, indices in z.items():
        H -= (len(indices) / y.shape[0]) * np.log2(len(indices) / y.shape[0])
    return H

def mutual_information(x, y):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """
    x_partition = partition(x)
    H_given = 0
    for key, indices in x_partition.items():
        y_given_x = np.array([])
        for i in indices:
            y_given_x = np.append(y_given_x, y[i])
        H_given += entropy(y_given_x)
    return entropy(y) - H_given

def select_best_features(n, X, y):
    """Selects the n best features of a dataset based on their mutual information with y

    Args:
        n (int): Number of features to choose
        X (np.array): Matrix of all features in dataset
        y (np.array): Labels for each example in dataset

    Returns:
        np.array: New feature matrix with only the n best features
    """
    I_hash = {}
    I = []
    for i in range(X.shape[1]):
        x = mutual_information(X[:,i], y)
        I_hash[x] = i
        I.append(x)
    I = sorted(I)[::-1]
    best_features = []
    for i in range(n):
        best_features.append(I_hash[I[i]])
    return X[:,best_features]


