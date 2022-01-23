import numpy as np
import pytest

import polynomial as poly


@pytest.mark.parametrize('degree, length', [(1, 2), (2, 3), (3, 4)])
def test_polynomial_create(degree, length):
    polynomial = poly.create(degree=degree)

    assert len(polynomial) == length


@pytest.mark.parametrize('polynomial, x, value', [
    (np.array([1]), 2, 1), 
    (np.array([4, 2, 1]), 2, 21), 
    (np.array([2, 4]), -2, 0)
])
def test_get_value(polynomial, x, value):
    assert poly.get_value(polynomial=polynomial, x=x) == value


@pytest.mark.parametrize('polynomial, derivative', [
    (np.array([4, 3, 2, 1]), np.array([12, 6, 2])),
    (np.array([1]), np.array([0])),
    (np.array([]), np.array([0])),
    (np.array([3, 2]), np.array([3])),
])
def test_get_derivative(polynomial, derivative):
    comparison = poly.get_derivative(polynomial=polynomial) == derivative

    assert comparison.all()

