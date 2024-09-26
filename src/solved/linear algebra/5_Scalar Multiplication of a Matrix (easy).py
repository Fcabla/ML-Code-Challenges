"""
problem_id: 5
Category: linear algebra
Difficulty: easy
URL: https://www.deep-ml.com/problem/Scalar%20Multiplication%20of%20a%20Matrix
Page: 3

==== Title ====
Scalar Multiplication of a Matrix (easy)

==== Description ====
Write a Python function that multiplies a matrix by a scalar and returns the result.

==== Example ====
Example:
        input: matrix = [[1, 2], [3, 4]], scalar = 2
        output: [[2, 4], [6, 8]]
        reasoning: Each element of the matrix is multiplied by the scalar.

==== Learn More ====
Scalar Multiplication of a Matrix

When a matrix \(A\) is multiplied by a scalar \(k\), the operation is defined as multiplying each element of \(A\) by \(k\).

Given a matrix \(A\):
\[
A = \begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{pmatrix}
\]

And a scalar \(k\), the result of the scalar multiplication \(kA\) is:
\[
kA = \begin{pmatrix}
ka_{11} & ka_{12} \\
ka_{21} & ka_{22}
\end{pmatrix}
\]

This operation scales the matrix by \(k\) without changing its dimension or the relative proportion of its elements.
"""


# ==== Code ====
def scalar_multiply(
    matrix: list[list[int | float]], scalar: int | float
) -> list[list[int | float]]:
    result = []
    for row in matrix:
        tmp = []
        for el in row:
            tmp.append(el * scalar)
        result.append(tmp)
    return result


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"matrix": [[1, 2], [3, 4]], "scalar": 2}
    print(inp)
    out = scalar_multiply(**inp)
    print(f"Output: {out}")
    exp = [[2, 4], [6, 8]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 1:")
    inp = {"matrix": [[0, -1], [1, 0]], "scalar": -1}
    print(inp)
    out = scalar_multiply(**inp)
    print(f"Output: {out}")
    exp = [[0, 1], [-1, 0]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
