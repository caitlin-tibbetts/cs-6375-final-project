import pytest
import numpy as np

from algorithms import naive_bayes

def test_categorize():
    X = np.array([[1,1,2],[3,3,1],[1,2,3],[3,3,3]])
    y = np.array([1,1,0,0])
    c = naive_bayes.categorize(X, y)
    assert np.array_equal(c[1], np.array([[1,1,2],[3,3,1]]))
    assert np.array_equal(c[0], np.array([[1,2,3],[3,3,3]]))

def test_fit():
    X = np.array([[1,1,2],[3,3,1],[1,2,3],[3,3,3]])
    y = np.array([1,1,0,0])
    f = naive_bayes.fit(X,y)
    print(f)
    assert f == {1:[(2, 1.41), (2, 1.41), (1.5, .71)], 0: [(2,.71),(2.5,.71),(3,0)]}