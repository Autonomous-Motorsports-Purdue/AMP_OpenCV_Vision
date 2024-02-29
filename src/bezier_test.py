import numpy as np
import matplotlib.pyplot as plt
from matplotlib import bezier
from math import factorial

def normalize_path_length(points):
    """
    Returns a list of the normalized path length of the points.
    """
    path_length = [0]
    x, y = points[:,0], points[:,1]

    # calculate the path length
    for i in range(1, len(points)):
        path_length.append(np.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2) + path_length[i - 1])
    
    # normalize the path length
    normal_length = []
    for i in range(len(path_length)):
        normal_length.append(path_length[i] / path_length[-1])
    
    return normal_length

def get_bezier(points):
    """
    Returns the control points of a bezier curve.
    """
    num_points = len(points)

    x, y = points[:,0], points[:,1]

    # bezier matrix for a cubic curve
    bezier_matrix = np.array([[-1, 3, -3, 1,], [3, -6, 3, 0], [-3, 3, 0, 0], [1, 0, 0, 0]])
    bezier_inverse = np.linalg.inv(bezier_matrix)

    normalized_length = normalize_path_length(points)

    points_matrix = np.zeros((num_points, 4))

    for i in range(num_points):
        points_matrix[i] = [normalized_length[i]**3, normalized_length[i]**2, normalized_length[i], 1]

    points_transpose = points_matrix.transpose()
    square_points = np.matmul(points_transpose, points_matrix)

    square_inverse = np.zeros_like(square_points)

    if (np.linalg.det(square_points) == 0):
        print("Uninvertible matrix")
        square_inverse = np.linalg.pinv(square_points)
    else:
        square_inverse = np.linalg.inv(square_points)

    # solve for the solution matrix
    solution = np.matmul(np.matmul(bezier_inverse, square_inverse), points_transpose)

    # solve for the control points
    control_points_x = np.matmul(solution, x)
    control_points_y = np.matmul(solution, y)

    return list(zip(control_points_x, control_points_y))

def comb(n, k):
    """
    Returns the combination of n choose k.
    """
    return factorial(n) / factorial(k) / factorial(n - k)

def plot_bezier(t, cp):
    """
    Plots a bezier curve.
    t is the time values for the curve.
    cp is the control points of the curve.
    return is a tuple of the x and y values of the curve.
    """
    cp = np.array(cp)
    num_points, d = np.shape(cp)   # Number of points, Dimension of points
    num_points = num_points - 1
    curve = np.zeros((len(t), d))
    
    for i in range(num_points+1):
        # Bernstein polynomial
        val = comb(num_points,i) * t**i * (1.0-t)**(num_points-i)
        curve += np.outer(val, cp[i])
    
    return curve

x = [0, 1, 2, 1]
y= [1, 4, 9, 10]

points = np.array([x, y]).transpose()

points = np.random.rand(5, 2)
points.sort(axis=0)

x, y = points[:,0], points[:,1]

t_vals = np.linspace(0, 1, 100)

print(points)

cp = np.array(get_bezier(points))

print(cp)

curve = plot_bezier(t_vals, cp)

plt.plot(curve[:,0], curve[:,1])
plt.plot(x, y, 'ro')
plt.plot(cp[:,0], cp[:,1], 'go')
plt.show()