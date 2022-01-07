import numpy as np
from typing import Tuple


def create(degree: int=2) -> np.ndarray:
    """Creates random polynomial of given degree


    Keyword arguments:
    degree -- degree of the polynomial

    low -- low boundary of coefficients range

    high -- high boundary of coefficients range

    Return: np.ndarray containing coefficients of created polynomial
    
    Returned array should be interpreted as a coefficients from 
    degree-1 to 0. Ie.:
    coeffs = np.ndarray([4, 5, 1])

    f(x) = 4 * x ^ 2 + 5 * x ^ 1 + 1 * x ^ 0

    """
    size = degree + 1
    coefs = np.random.random_sample(size=size) * np.linspace(10, 1, num=size) + np.linspace(1, 0, size)
    return coefs

def get_values_in_domain(
        polynomial: np.ndarray, 
        low: int=-5, 
        high: int=5, 
        probes: int=1000) -> Tuple[np.ndarray, np.ndarray]:
    """Gets values of given polynomial in given (low, high) domain (low and high included)
    
    Keyword arguments:
    polynomial -- np.ndarray with coefficients describing this polynomial
    Return: Tuple containing (values, domain)
    """
    step = abs(low - high) / probes

    values = np.zeros(shape=probes+1, dtype=np.float32)
    domain = np.linspace(start=low, stop=high, num=probes + 1)

    x = low
    id = 0

    for x in domain:
        values[id] = get_value(polynomial, x)
        x += step
        id += 1

    return values, domain

def get_value(polynomial: np.ndarray, x: np.float32) -> np.float32:
    """Calculates value of given polynomial in point x

    
    Keyword arguments:
    polynomial -- np.ndarray with coefficients describing this polynomial

    x -- np.float32 x value to calculate polynomial value

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
    polynomial -- polynomial in form of np.ndarray with coefficients in order from 
    degree-1 to 0. Ie. [4, 5, 1] -> f(x) = 4 * x ^ 2 + 5 * x ^ 1 + 1 * x ^ 0

    Then derivative of [4, 5, 1] will be [8, 5] -> f'(x) = 8 * x + 5

    Return: derivative as np.ndarray of coefficients
    """
    
    size = polynomial.shape[0]

    if size < 2:
        return np.zeros(shape=1)

    derivative = polynomial[:-1:] * np.arange(start=size-1, stop=0, step=-1)

    return derivative

def get_multivariable_polynomial_values(poly_1: np.ndarray, poly_2: np.ndarray, domain: Tuple[int, int]) -> np.ndarray:
    """Returns matrix representing values of sum of the two given polynomials
    
    Keyword arguments:
    poly_1, poly_2  --  polynomials to get value from
    domain  --  tuple of domain boundaries


    Return: 2 dimensional np.ndarray with sum of these polynomials in given domain
    """
    domain_low, domain_high = domain
    
    values_1, _ = get_values_in_domain(
        polynomial=poly_1, low=domain_low, high=domain_high)
    values_2, _ = get_values_in_domain(
        polynomial=poly_2, low=domain_low, high=domain_high)
    
    Z = np.zeros(shape=(1001, 1001))

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = values_1[i] + values_2[j]

    return Z

def get_multivariable_polynomial_value(poly_1, poly_2, x, y):

    return get_value(poly_1, x) + get_value(poly_2, y)
