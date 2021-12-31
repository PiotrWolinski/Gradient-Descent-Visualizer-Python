from matplotlib import pyplot as plt
from matplotlib import cm
import numpy as np
import polynomial as poly 

def plot_3D(f_x, f_y):

    ax = plt.axes(projection='3d')

    X = np.linspace(-100, 100, 1001)
    Y = np.linspace(-100, 100, 1001)
    X, Y = np.meshgrid(X, Y)

    Z = np.zeros(shape=(1001, 1001))

    for i in range(Z.shape[0]):
        for j in range(Z.shape[1]):
            Z[i, j] = f_x[i] + f_y[j]


    ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    plt.show()

def main():
    f1 = poly.create(degree=2)
    f2 = poly.create(degree=2)
    print(f1)
    print(f2)
    f1_y, x = poly.get_values_in_domain(polynomial=f1)
    f2_y, y = poly.get_values_in_domain(polynomial=f2)

    print(f1_y)
    print(f2_y)
    plot_3D(f1_y, f2_y)

if __name__ == '__main__':
    main()
    