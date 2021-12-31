from matplotlib import pyplot as plt
import numpy as np
import polynomial as poly 

def plot_3D(f_x, f_y, f_z):

    f = plt.figure()

    ax = plt.axes(projection='3d')
    x = f_x.tolist()
    # print(x)
    y = f_y.tolist()
    # print(y)
    z = f_z.tolist()
    ax.scatter(x, y, z)
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.label('z')
    # plt.title('Time comparison between Jacobi and Gauss-Seidl methods')
    # plt.legend()
    plt.show()

def main():
    f1 = poly.create(degree=4)
    f2 = poly.create(degree=4)
    f3 = poly.create(degree=4)
    print(f1)
    print(f2)
    print(f3)
    f1_y, x = poly.get_values_in_domain(polynomial=f1)
    f2_y, y = poly.get_values_in_domain(polynomial=f2)
    f3_y, z = poly.get_values_in_domain(polynomial=f3)

    print(f1_y)
    print(f2_y)
    print(f3_y)
    plot_3D(f1_y, f2_y, f3_y)

if __name__ == '__main__':
    main()