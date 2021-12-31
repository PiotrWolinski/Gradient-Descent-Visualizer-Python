import numpy as np


def create(degree: int=2, low: int=-10, high: int=10) -> np.ndarray:
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
    coefs = np.random.randint(low=low, high=high, size=size, dtype=np.int32)
    return coefs

def get_values_in_domain(polynomial: np.ndarray, low: int=-5, high: int=5, probes: int=1000) -> np.int32:
    """Gets values of given polynomial in given (low, high) domain (low and high included)
    
    Keyword arguments:
    polynomial -- np.ndarray with coefficients describing this polynomial
    Return: return_description
    """
    domain_size = abs(low - high) 

    step = abs(low - high) / probes

    values = np.zeros(shape=probes+1, dtype=np.float32)
    domain = np.arange(start=low, stop=high+step, step=step)

    x = low
    id = 0

    print(f'step = {step}')
    print(f'size = {domain_size}')

    for x in domain:
        values[id] = get_value(polynomial, x)
        x += step
        id += 1

    print(values.shape)
    print(domain.shape)

    return values, domain

def get_value(polynomial: np.ndarray, x: np.float32):
    """Calculates value of given polynomial in point x

    
    Keyword arguments:
    polynomial -- np.ndarray with coefficients describing this polynomial

    x -- np.float32 x value to calculate polynomial value


    Return: np.float32 value of given polynomial in point x 
    """
    
    value = 0.0

    degree = polynomial.shape[0] - 1
    current_id = 0

    while current_id <= degree:
        value += polynomial[current_id] * x ** (degree - current_id)
        current_id += 1

    return value
