from math import ceil, floor
from typing import List, Tuple

import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt

import polynomial as poly


def plot_3D_gradient_descent(
        f_x: np.ndarray, 
        f_y: np.ndarray, 
        descent_path: List[float], 
        domain: np.ndarray, 
        values_per_unit: int) -> None:

    ax = plt.axes(projection='3d')

    low = floor(domain[0])
    high = ceil(domain[1])

    domain_size = (high - low) * values_per_unit + 1
    print(domain_size)

    X = np.linspace(low, high, domain_size)
    Y = np.linspace(low, high, domain_size)
    X, Y = np.meshgrid(X, Y)

    Z = np.zeros(shape=(domain_size, domain_size))

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = poly.get_multivariable_polynomials_sum(
                f_x, f_y, X[i, j], Y[i, j])

    descent_x = [x[0] for x in descent_path]
    descent_y = [x[1] for x in descent_path]
    descent_z = [x[2] for x in descent_path]

    ax.scatter(descent_x, descent_y, descent_z, color=['red'], s=50)
    ax.plot(descent_x, descent_y, descent_z, 'r')

    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False, alpha=0.4)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()


def create_descent_path(
        poly_x: np.ndarray,
        poly_y: np.ndarray,
        domain: np.ndarray,
        gradient_weight: float,
        steps: int) -> np.ndarray:

    start_x = domain[0] + np.random.rand() * (domain[-1] - domain[0])
    start_y = domain[0] + np.random.rand() * (domain[-1] - domain[0])

    start_value = poly.get_multivariable_polynomials_sum(
        poly_x, poly_y, start_x, start_y)

    derivative_x = poly.get_derivative(poly_x)
    derivative_y = poly.get_derivative(poly_y)

    descent_path = [(start_x, start_y, start_value)]

    current_x = start_x
    current_y = start_y

    for _ in range(steps):
        gradient_x = poly.get_value(derivative_x, current_x)
        gradient_y = poly.get_value(derivative_y, current_y)

        current_x -= gradient_weight * gradient_x
        current_y -= gradient_weight * gradient_y

        descent_path.append((current_x, current_y, poly.get_multivariable_polynomials_sum(
            poly_x, poly_y, current_x, current_y)))

    return descent_path


def validate_domain(domain):
    """This functions validates if given domain meets the criteria
        for the nice graph plotting.

        Criteria:
        type(domain) == Tuple[float] or Tuple[int]
        len(domain)  == 2
        domain[0] < domain[1]
        abs(domain[1] - domain[0]) > 10
    
    Raises error if domain given does not meet this criteria.
    """
    
    assert len(domain) == 2
    assert type(domain) is tuple
    assert type(domain[0]) is int and type(domain[1]) is int \
        or type(domain[0]) is float and type(domain[1]) is float
    assert domain[0] < domain[1]
    assert abs(domain[1] - domain[0]) > 1


def main() -> None:
    
    # Specify polynomials that will describe the surface
    f_x = poly.create(degree=2)
    f_y = poly.create(degree=2)

    # Specify domain in which surface with created path
    # should be plotted. 
    domain = (-2, 2)

    # Check if domain given meets the criteria for a nice graph
    validate_domain(domain)

    # The highest the value, the more precise grpah will be
    values_per_unit = 10

    # Set the weight of the gradient - it tells how much to relly 
    # on the gradient - set this value too high, and path will be unable
    # to converge to the minimum, too low and it will go there to slow.
    # For me 0.02 was a sweet spot
    gradient_weight = 0.02

    # Amount of steps in the descent path
    steps = 5

    # Create descent path based on the variables created eariler
    descent_path = create_descent_path(
        f_x, f_y, domain, gradient_weight, steps)

    # Plot surface with given descent path.
    plot_3D_gradient_descent(f_x, f_y, descent_path, domain, values_per_unit)


if __name__ == '__main__':
    main()
