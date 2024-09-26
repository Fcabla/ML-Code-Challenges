"""
problem_id: 7
Category: linear algebra
Difficulty: medium
URL: https://www.deep-ml.com/problem/Matrix%20Transformation
Page: 3

==== Title ====
Matrix Transformation (medium)

==== Description ====
Write a Python function that transforms a given matrix A using the operation $T^{-1}AS$, where T and S are invertible matrices. The function should first validate if the matrices T and S are invertible, and then perform the transformation. In cases where there is no solution return -1

==== Example ====
Example:
        input: A = [[1, 2], [3, 4]], T = [[2, 0], [0, 2]], S = [[1, 1], [0, 1]]
        output: [[0.5,1.5],[1.5,3.5]]
        reasoning: The matrices T and S are used to transform matrix A by computing $T^{-1}AS$.

==== Learn More ====
Matrix Transformation using \(T^{-1}AS\)

Transforming a matrix \(A\) using the operation \(T^{-1}AS\) involves several steps. This operation changes the basis of matrix \(A\) using two invertible matrices \(T\) and \(S\).

Given matrices \(A\), \(T\), and \(S\):

1. Check if \(T\) and \(S\) are invertible by ensuring their determinants are non-zero; else return -1.
2. Compute the inverses of \(T\) and \(S\), denoted as \(T^{-1}\) and \(S^{-1}\).
3. Perform the matrix multiplication to obtain the transformed matrix:
\[
A' = T^{-1}AS
\]

 Example
If:
\[
A = \begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix}
\]
\[
T = \begin{pmatrix}
2 & 0 \\
0 & 2
\end{pmatrix}
\]
\[
S = \begin{pmatrix}
1 & 1 \\
0 & 1
\end{pmatrix}
\]

First, check that \(T\) and \(S\) are invertible:
• \(\det(T) = 4 \neq 0 \)
• \(\det(S) = 1 \neq 0 \)

Compute the inverses:
\[
T^{-1} = \begin{pmatrix}
\frac{1}{2} & 0 \\
0 & \frac{1}{2}
\end{pmatrix}
\]

Then, perform the transformation:
\[
A' = T^{-1}AS = \begin{pmatrix}
\frac{1}{2} & 0 \\
0 & \frac{1}{2}
\end{pmatrix} \begin{pmatrix}
1 & 2 \\
3 & 4
\end{pmatrix} \begin{pmatrix}
1 & 1 \\
0 & 1
\end{pmatrix} = \begin{pmatrix}
0.5 & 1.5 \\
1.5 & 3.5
\end{pmatrix}
\]
"""

# ==== Code ====
import numpy as np


def transform_matrix(
    A: list[list[int | float]], T: list[list[int | float]], S: list[list[int | float]]
) -> list[list[int | float]]:
    A = np.array(A, dtype=float)
    T = np.array(T, dtype=float)
    S = np.array(S, dtype=float)

    # Check invertibility
    if np.linalg.det(T) == 0:
        return -1
    if np.linalg.det(S) == 0:
        return -1

    inverseT = np.linalg.inv(T)
    transformed_matrix = np.matmul(np.matmul(inverseT, A), S)
    return np.round(transformed_matrix, 8).tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"A": [[1, 2], [3, 4]], "T": [[2, 0], [0, 2]], "S": [[1, 1], [0, 1]]}
    print(inp)
    out = transform_matrix(**inp)
    print(f"Output: {out}")
    exp = [[0.5, 1.5], [1.5, 3.5]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"A": [[1, 0], [0, 1]], "T": [[1, 2], [3, 4]], "S": [[2, 0], [0, 2]]}
    print(inp)
    out = transform_matrix(**inp)
    print(f"Output: {out}")
    exp = [[-4.0, 2.0], [3.0, -1.0]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {"A": [[2, 3], [1, 4]], "T": [[3, 0], [0, 3]], "S": [[1, 1], [0, 1]]}
    print(inp)
    out = transform_matrix(**inp)
    print(f"Output: {out}")
    exp = [[0.66666667, 1.66666667], [0.33333333, 1.66666667]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 4
    print("Test Case 4:")
    inp = {"A": [[2, 3], [1, 4]], "T": [[3, 0], [0, 3]], "S": [[1, 1], [1, 1]]}
    print(inp)
    out = transform_matrix(**inp)
    print(f"Output: {out}")
    exp = -1
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
