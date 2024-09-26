"""
problem_id: 8
Category: linear algebra
Difficulty: medium
URL: https://www.deep-ml.com/problem/Calculate%202x2%20Matrix%20Inverse
Page: 3

==== Title ====
Calculate 2x2 Matrix Inverse (medium)

==== Description ====
Write a Python function that calculates the inverse of a 2x2 matrix. Return 'None' if the matrix is not invertible.

==== Example ====
Example:
        input: matrix = [[4, 7], [2, 6]]
        output: [[0.6, -0.7], [-0.2, 0.4]]
        reasoning: The inverse of a 2x2 matrix [a, b], [c, d] is given by (1/(ad-bc)) * [d, -b], [-c, a], provided ad-bc is not zero.

==== Learn More ====
Calculating the Inverse of a 2x2 Matrix

The inverse of a matrix \(A\) is another matrix, often denoted \(A^{-1}\), such that:
\[
AA^{-1} = A^{-1}A = I
\]
where \(I\) is the identity matrix. For a 2x2 matrix:
\[
A = \begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\]

The inverse is:
\[
A^{-1} = \frac{1}{\det(A)} \begin{pmatrix}
d & -b \\
-c & a
\end{pmatrix}
\]

provided that the determinant \(\det(A) = ad - bc\) is non-zero. If \(\det(A) = 0\), the matrix does not have an inverse.

This process is critical in many applications including solving systems of linear equations, where the inverse is used to find solutions efficiently.
"""


# ==== Code ====
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    determinant = (a * d) - (b * c)
    if determinant == 0:
        return None
    tmp = 1 / (a * d - b * c)
    inverse = [[d * tmp, -b * tmp], [-c * tmp, a * tmp]]
    # round
    inverse = [
        [round(inverse[i][j], 1) for j in range(len(inverse[i]))]
        for i in range(len(inverse))
    ]
    return inverse


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"matrix": [[4, 7], [2, 6]]}
    print(inp)
    out = inverse_2x2(**inp)
    print(f"Output: {out}")
    exp = [[0.6, -0.7], [-0.2, 0.4]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"matrix": [[2, 1], [6, 2]]}
    print(inp)
    out = inverse_2x2(**inp)
    print(f"Output: {out}")
    exp = [[-1.0, 0.5], [3.0, -1.0]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
