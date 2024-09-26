"""
problem_id: 11
Category: linear algebra
Difficulty: medium
URL: https://www.deep-ml.com/problem/Solve%20Linear%20Equations%20using%20Jacobi%20Method
Page: 1

==== Title ====
Solve Linear Equations using Jacobi Method (medium)

==== Description ====
Write a Python function that uses the Jacobi method to solve a system of linear equations given by Ax = b. The function should iterate 10 times, rounding each intermediate solution to four decimal places, and return the approximate solution x.

==== Example ====
Example:
        input: A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]], b = [-1, 2, 3], n=2
        output: [0.146, 0.2032, -0.5175]
        reasoning: The Jacobi method iteratively solves each equation for x[i] using the formula x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i)), where a_ii is the diagonal element of A and a_ij are the off-diagonal elements.

==== Learn More ====
Solving Linear Equations Using the Jacobi Method

The Jacobi method is an iterative algorithm used for solving a system of linear equations \(Ax = b\). This method is particularly useful for large systems where direct methods like Gaussian elimination are computationally expensive.

 Algorithm Overview

For a system of equations represented by \(Ax = b\), where \(A\) is a matrix and \(x\) and \(b\) are vectors, the Jacobi method involves the following steps:

Initialization: Start with an initial guess for \(x\).
Iteration: For each equation \(i\), update \(x[i]\) using:
   \[
   x[i] = \frac{1}{a_{ii}} (b[i] - \sum_{j \neq i} a_{ij} x[j])
   \]
   where \(a_{ii}\) are the diagonal elements of \(A\), and \(a_{ij}\) are the off-diagonal elements.
Convergence: Repeat the iteration until the changes in \(x\) are below a certain tolerance or until a maximum number of iterations is reached.

This method assumes that all diagonal elements of \(A\) are non-zero and that the matrix is diagonally dominant or properly conditioned for convergence.

 Practical Considerations

- The method may not converge for all matrices.
- Choosing a good initial guess can improve convergence.
- Diagonal dominance of \(A\) ensures convergence of the Jacobi method.
"""

# ==== Code ====
import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = [0 for _ in b]
    # x[i] using the formula x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i)),
    # where a_ii is the diagonal element of A and a_ij are the off-diagonal elements.
    for iteration in range(n):
        # hold previous values
        x_new = x.copy()
        for i in range(len(A)):
            suma = 0
            for j in range(len(A)):
                if i != j:
                    suma += A[i][j] * x[j]
            x_new[i] = (1 / A[i][i]) * (b[i] - suma)

        x = x_new
    return np.round(x, 4).tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "A": np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]]),
        "b": np.array([-1, 2, 3]),
        "n": 2,
    }
    print(inp)
    out = solve_jacobi(**inp)
    print(f"Output: {out}")
    exp = [0.146, 0.2032, -0.5175]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "A": np.array([[4, 1, 2], [1, 5, 1], [2, 1, 3]]),
        "b": np.array([4, 6, 7]),
        "n": 5,
    }
    print(inp)
    out = solve_jacobi(**inp)
    print(f"Output: {out}")
    exp = [-0.0806, 0.9324, 2.4422]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {
        "A": np.array([[4, 2, -2], [1, -3, -1], [3, -1, 4]]),
        "b": np.array([0, 7, 5]),
        "n": 3,
    }
    print(inp)
    out = solve_jacobi(**inp)
    print(f"Output: {out}")
    exp = [1.7083, -1.9583, -0.7812]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
