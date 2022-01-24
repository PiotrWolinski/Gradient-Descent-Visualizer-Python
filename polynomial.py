import numpy as np


def create(degree: int=2) -> np.ndarray:
    """Creates random polynomial of given degree

    Keyword arguments:
    degree: degree of the polynomial

    low: low boundary of coefficients range

    high: high boundary of coefficients range

    Returned array should be interpreted as a coefficients from 
    degree-1 to 0. Ie.:
    coeffs = np.ndarray([4, 5, 1])

    f(x) = 4 * x ^ 2 + 5 * x ^ 1 + 1 * x ^ 0

    Return: np.ndarray containing coefficients of created polynomial
    """
    size = degree + 1

    # linspace is added to emphasize the coefficent of the highest degree,
    # so that the slopes of the graph would be nice
    coefs = np.random.random_sample(size=size) * np.linspace(10, 1, num=size) + np.linspace(1, 0, size)

    return coefs

def get_value(polynomial: np.ndarray, x: np.float32) -> np.float32:
    """Calculates value of given polynomial in point x
    
    Keyword arguments:
    polynomial: np.ndarray with coefficients describing this polynomial

    x: np.float32 x value to calculate polynomial value

    Return: np.float32 value of given polynomial in point x 
    """
    value = np.float32(0.0)

    degree = polynomial.shape[0] - 1
    current_id = 0

    while current_id <= degree:
        value += polynomial[current_id] * x ** (degree - current_id)
        current_id += 1

    return value


def get_derivative(polynomial: np.ndarray) -> np.ndarray:
    """Calculates derivative of given polynomial
    
    Keyword arguments:
    polynomial: polynomial in form of np.ndarray with coefficients in order from 
        degree-1 to 0. Ie. [4, 5, 1] -> f(x) = 4 * x ^ 2 + 5 * x ^ 1 + 1 * x ^ 0

    Then derivative of [4, 5, 1] will be [8, 5] -> f'(x) = 8 * x + 5

    Return: derivative as np.ndarray of coefficients
    """
    
    size = polynomial.shape[0]

    if size < 2:
        return np.zeros(shape=1)

    derivative = polynomial[:-1:] * np.arange(start=size-1, stop=0, step=-1)

    return derivative


def get_multivariable_polynomials_sum(poly_1: np.ndarray, poly_2: np.ndarray, x: np.float32, y: np.float32) -> np.float32:
    """Returns sum of two polynomials of different variables.
    
    Keyword arguments:
    poly_1, poly_2: np.ndarrays of coefficients describing polynomials
    x, y: np.float32 variables to calculate values of the polynomials
    Return: np.float32 sum of the polynomials values for the given variables
    """
    
    return get_value(poly_1, x) + get_value(poly_2, y)
