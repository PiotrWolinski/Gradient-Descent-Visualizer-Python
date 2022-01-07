from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import polynomial as poly
from typing import Tuple


def plot_3D(polynomial_values, descent_path, domain) -> None:

    ax = plt.axes(projection='3d')

    low = domain[0]
    high = domain[1]

    X = np.linspace(low, high, 1001)
    Y = np.linspace(low, high, 1001)
    X, Y = np.meshgrid(X, Y)

    descent_x = [x[0] for x in descent_path]
    descent_y = [x[1] for x in descent_path]
    descent_z = [x[2] for x in descent_path]

    ax.scatter(descent_x, descent_y, descent_z, color=['red'], s=80)
    ax.plot(descent_x, descent_y, descent_z, 'r')

    ax.plot_surface(X, Y, polynomial_values, cmap=cm.coolwarm,
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
        iterations: int) -> np.ndarray:
    derivative_x = poly.get_derivative(poly_x)
    derivative_y = poly.get_derivative(poly_y)

    start_x = domain[0] + np.random.rand() * (domain[-1] - domain[0])
    start_y = domain[0] + np.random.rand() * (domain[-1] - domain[0])

    polynomials_values = poly.get_multivariable_polynomial_values(poly_x, poly_y, domain)

    start_value = poly.get_multivariable_polynomial_value(poly_x, poly_y, start_x, start_y)

    descent_path = [(start_x, start_y, start_value)]

    current_x = start_x
    current_y = start_y

    print(f'({current_x}, {current_y}')
    for i in range(iterations):
        # gradient = gradient_weight * poly.get_multivariable_polynomial_value(derivative_x, derivative_y, current_x, current_y)
        gradient_x = poly.get_value(derivative_x, current_x)
        gradient_y = poly.get_value(derivative_y, current_y)

        current_x -= gradient_weight * gradient_x
        current_y -= gradient_weight * gradient_y

        descent_path.append((current_x, current_y, poly.get_multivariable_polynomial_value(poly_x, poly_y, current_x, current_y)))
    
        # print(gradient_x)
        # print(gradient_y)

        print(f'G=[{gradient_x}, {gradient_y}')

        print(f'({current_x}, {current_y})')

    print(descent_path)

    return descent_path

def main() -> None:
    f_x = poly.create(degree=2)
    f_y = poly.create(degree=2)

    # print(f'f_x = {f_x}')
    # print(f'f_y = {f_y}')

    domain = (-5, 5)

    polynomials_values = poly.get_multivariable_polynomial_values(f_x, f_y, domain)

    gradient_weight = 0.01
    iterations = 5

    descent_path = create_descent_path(f_x, f_y, domain, gradient_weight, iterations)

    plot_3D(polynomials_values, descent_path, domain)


if __name__ == '__main__':
    main()
