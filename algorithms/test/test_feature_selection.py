import pytest
import numpy as np

from algorithms import feature_selection

def test_partition():
    y = np.array([1,1,1,1,1,1,1,1,1,1,0,0])
    p = feature_selection.partition(y)
    assert p == {1:[0,1,2,3,4,5, 6,7,8,9], 0:[10,11]}

def test_partition_empty():
    y = np.array([])
    p = feature_selection.partition(y)
    assert p == {}

def test_entropy():
    y = np.array([1,1,1,1,1,1,1,1,1,1,0,0])
    h = feature_selection.entropy(y)
    assert round(h,2) == .65

def test_mutual_information():
    x = np.array([3,3,3,2,2,3,0,0,0,0,0,0])
    y = np.array([1,1,1,1,1,1,1,1,1,1,0,0])
    i = feature_selection.mutual_information(x,y)
    assert round(i,2) == -.27