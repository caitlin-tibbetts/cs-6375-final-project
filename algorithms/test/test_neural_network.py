import numpy as np

from algorithms import neural_network


def test_init_params():
    params = neural_network.init_params([3, 2, 1])
    assert all([x in params for x in ["W1", "W2", "b1", "b2"]])


def test_forward_propagation():
    output = neural_network.forward_propagation(
        [[1, 1, 1, 2, 2], [1, 2, 1, 2, 1], [2, 2, 1, 2, 1]],
        neural_network.init_params([3, 2, 2, 1]),
        lambda x: np.maximum(0, x),
    )
    assert output.shape[1] == 5