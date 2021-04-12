import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


def init_params(layers):
    """Initializes all of the weights and biases for each layer of the neural network

    Args:
        layers (list): Number of nodes in each layer of the neural network

    Returns:
        Dict: Each entry is the weights and biases of the nodes given the layer
    """
    model_params = {}
    for i in range(1, len(layers)):
        model_params["W" + str(i)] = np.random.randn(layers[i], layers[i - 1]) * 0.01
        model_params["b" + str(i)] = np.random.randn(layers[i], 1) * 0.01
    return model_params


def forward_propagation(input, model_params, activation, cur_layer=1):
    input = (
        np.dot(model_params["W" + str(cur_layer)], input)
        + model_params["b" + str(cur_layer)]
    )
    if cur_layer == len(model_params) // 2:
        return input
    else:
        return forward_propagation(
            activation(input), model_params, activation, cur_layer + 1
        )


def compute_cost(X, y):
    y_pred = forward_propagation(
        X.T, init_params([X.shape[1], 5, 2, 1]), lambda x: np.maximum(0, x)
    )
    cost = (1 / (2 * len(y))) * np.sum(np.square(y_pred - y))
    return cost


if __name__ == "__main__":
    dataset = load_boston()
    X = dataset["data"]
    y = dataset["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, test_size=0.2, random_state=20
    )

    print(compute_cost(X_train, y_train))
