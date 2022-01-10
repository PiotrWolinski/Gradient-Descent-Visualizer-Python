from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import polynomial as poly
from typing import Tuple


def plot_3D_gradient_descent(f_x, f_y, descent_path, domain) -> None:

    ax = plt.axes(projection='3d')

    low = domain[0]
    high = domain[1]

    X = np.linspace(low, high, 1001)
    Y = np.linspace(low, high, 1001)
    X, Y = np.meshgrid(X, Y)

    Z = np.zeros(shape=(1001, 1001))

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
        iterations: int) -> np.ndarray:

    start_x = domain[0] + np.random.rand() * (domain[-1] - domain[0])
    start_y = domain[0] + np.random.rand() * (domain[-1] - domain[0])

    start_value = poly.get_multivariable_polynomials_sum(
        poly_x, poly_y, start_x, start_y)

    derivative_x = poly.get_derivative(poly_x)
    derivative_y = poly.get_derivative(poly_y)

    descent_path = [(start_x, start_y, start_value)]

    current_x = start_x
    current_y = start_y

    for _ in range(iterations):
        gradient_x = poly.get_value(derivative_x, current_x)
        gradient_y = poly.get_value(derivative_y, current_y)

        current_x -= gradient_weight * gradient_x
        current_y -= gradient_weight * gradient_y

        descent_path.append((current_x, current_y, poly.get_multivariable_polynomials_sum(
            poly_x, poly_y, current_x, current_y)))

    return descent_path


def main() -> None:
    f_x = poly.create(degree=2)
    f_y = poly.create(degree=2)

    domain = (-5, 5)

    gradient_weight = 0.01
    iterations = 5

    descent_path = create_descent_path(
        f_x, f_y, domain, gradient_weight, iterations)

    plot_3D_gradient_descent(f_x, f_y, descent_path, domain)


if __name__ == '__main__':
    main()
