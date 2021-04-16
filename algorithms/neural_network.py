import numpy as np


def init(layers):
    """Initializes all of the weights and biases for each layer of the neural network

    Args:
        layers (list): Number of nodes in each layer of the neural network

    Returns:
        Dict: Each entry is the weights and biases of the nodes given the layer
    """
    W, b = np.array([]), np.array([])
    for i in range(1, len(layers)):
        W = np.append(W, np.random.randn(layers[i], layers[i - 1]) * 0.01)
        b = np.append(b, np.random.randn(layers[i], 1) * 0.01)
    return W, b


def forward_propagation(
    input, W, b, activation=lambda x: np.maximum(0, x), cur_layer=1
):
    """Walks forward through the neural network, recursively, and returns the final output

    Args:
        input (np.array): Output of the last layer, starts as features of the dataset.
        W (np.array): Weights of every node connection in the neural network.
        b (np.array): Biases of every layer in the neural network.
        activation (function, optional): Chosen activation function. Defaults to ReLu.
        cur_layer (int, optional): Current layer the forward propagation is evaluating. Defaults to 1.

    Returns:
        np.array: Output of the neural network for the given data
    """
    input = np.dot(W[cur_layer], input) + b[cur_layer]

    if cur_layer == len(W) // 2:
        # Without calling the activation function because this is the output
        return input
    else:
        return forward_propagation(activation(input), W, b, activation, cur_layer + 1)


def compute_cost(X, y, W, b):
    """Computing the cost of a run through the neural network

    Args:
        X (np.array): Features of the dataset
        y (np.array): Classes of the dataset
        W (np.array): Weights of every node connection in the neural network.
        b (np.array): Biases of every layer in the neural network.
    Returns:
        float: The cost of the current neural network parameters
    """
    y_pred = forward_propagation(X.T, W, b)
    # Calulate the mean-squared error
    cost = (1 / (2 * len(y))) * np.sum(np.square(y_pred - y))
    return cost


def backward_propagation(
    X,
    y,
    W,
    b,
    A,
    dA=np.array([]),
    W_grads=np.array([]),
    b_grads=np.array([]),
    cur_layer=-1,
):
    """Performs backward propagation on the neural network to get the gradient of the cost in order to later better tune the weights and biases

    Args:
        X (np.array): Features of the dataset
        y (np.array): Classes of the dataset
        W (np.array): Weights of every node connection in the neural network.
        b (np.array): Biases of every layer in the neural network.
        A (np.array): Values of each node for each example
        dA (np.array, optional): Change in the value for each node. Defaults to np.array([]).
        W_grads (np.array, optional): Gradients for the weights. Defaults to np.array([]).
        b_grads (np.array, optional): Gradients for the biases. Defaults to np.array([]).
        cur_layer (int, optional): Current layer being evaluated. Defaults to -1.

    Returns:
        Tuple: The gradients of the weights and biases
    """
    if cur_layer == -1:
        # Just started the algorithm
        cur_layer = len(W) // 2
        dA = 1 / len(y) * (A[cur_layer] - y)
    else:
        dA = np.multiply(
            np.dot(W[cur_layer + 1].T, dA), np.where(A[cur_layer] >= 0, 1, 0)
        )

    if cur_layer == 1:
        W_grads[cur_layer] = 1 / len(y) * np.dot(dA, X_train.T)
        b_grads[cur_layer] = 1 / len(y) * np.sum(dA, axis=1, keepdims=True)
        return W_grads, b_grads
    else:
        W_grads[cur_layer] = 1 / len(y) * np.dot(dA, A[cur_layer - 1].T)
        b_grads[cur_layer] = 1 / len(y) * np.sum(dA, axis=1, keepdims=True)
        return backward_propagation(X, y, W, b, A, dA, W_grads, b_grads, cur_layer - 1)


def update(W, b, W_grads, b_grads, learning_rate):
    layers = len(W)//2
    W_updated = np.array([])
    b_updated = np.array([])
    for i in range(1,layers+1):
        W_updated[i] = W[i] - learning_rate * W_grads[i]
        b_updated[i] = b[i] - learning_rate * b_grads[i]
    return W_updated, b_updated

def predict(X, W, b):
    values = forward_propagation(X.T, W, b)
    return A[len(values)//2].T

def model(X_train, y_train, layers, num_iters, learning_rate):
    W, b = init(layers)
    for i in range(num_iters):
        y_pred = forward_propagation(X_train.T, W, b)
        cost = compute_cost(y_train.T, W, b)
        W_grads, b_grads = backward_propagation(X_train.T, y_train.T, W, b, y_pred.T)
        params = update(W, b, W_grads, b_grads, learning_rate)
        print('Cost at iteration ' + str(i+1) + ' = ' + str(cost) + '\n')
    return W, b