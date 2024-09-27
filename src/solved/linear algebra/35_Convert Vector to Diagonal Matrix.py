"""
problem_id: 35
Category: linear algebra
Difficulty: easy
URL: https://www.deep-ml.com/problem/Convert%20Vector%20to%20Diagonal%20Matrix
Page: 2

==== Title ====
Convert Vector to Diagonal Matrix

==== Description ====
Write a Python function to convert a 1D numpy array into a diagonal matrix. The function should take in a 1D numpy array x and return a 2D numpy array representing the diagonal matrix.

==== Example ====
Example:
    x = np.array([1, 2, 3])
    output = make_diagonal(x)
    print(output)
    # Output:
    # [[1. 0. 0.]
    #  [0. 2. 0.]
    #  [0. 0. 3.]]

    Reasoning:
    The input vector [1, 2, 3] is converted into a diagonal matrix where the elements of the vector form the diagonal of the matrix.

==== Learn More ====
Understanding Diagonal Matrices
A diagonal matrix is a square matrix in which the entries outside the main diagonal are all zero. The main diagonal is the set of entries extending from the top left to the bottom right of the matrix.
In this problem, you will write a function to convert a 1D numpy array (vector) into a diagonal matrix. The resulting matrix will have the elements of the input vector on its main diagonal, and zeros elsewhere.
Given a vector \( \mathbf{x} = [x_1, x_2, \ldots, x_n] \), the corresponding diagonal matrix \( \mathbf{D} \) is:

\[ \mathbf{D} = \begin{bmatrix}
x_1 & 0 & 0 & \cdots & 0 \\
0 & x_2 & 0 & \cdots & 0 \\
0 & 0 & x_3 & \cdots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & \cdots & x_n
\end{bmatrix} \]

Diagonal matrices are important in various mathematical and scientific computations because of their simple structure and properties.
"""

# ==== Code ====
import numpy as np


def make_diagonal(x):
    # Your code here
    n = len(x)
    result = np.zeros((n, n))
    for i in range(n):
        result[i, i] = x[i]
    return result.tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"x": np.array([1, 2, 3])}
    print(inp)
    out = make_diagonal(**inp)
    print(f"Output: {out}")
    exp = [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"x": np.array([4, 5, 6, 7])}
    print(inp)
    out = make_diagonal(**inp)
    print(f"Output: {out}")
    exp = [
        [4.0, 0.0, 0.0, 0.0],
        [0.0, 5.0, 0.0, 0.0],
        [0.0, 0.0, 6.0, 0.0],
        [0.0, 0.0, 0.0, 7.0],
    ]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
