"""
problem_id: 2
Category: linear algebra
Difficulty: easy
URL: https://www.deep-ml.com/problem/Transpose%20of%20a%20Matrix
Page: 1

==== Title ====
Transpose of a Matrix (easy)

==== Description ====
Write a Python function that computes the transpose of a given matrix.

==== Example ====
Example:
        input: a = [[1,2,3],[4,5,6]]
        output: [[1,4],[2,5],[3,6]]
        reasoning: The transpose of a matrix is obtained by flipping rows and columns.

==== Learn More ====
Transpose of a Matrix

Consider a matrix \(M\) and its transpose \(M^T\), where:

Original Matrix \(M\):
\[
M = \begin{pmatrix}
a & b & c \\
d & e & f
\end{pmatrix}
\]

Transposed Matrix \(M^T\):
\[
M^T = \begin{pmatrix}
a & d \\
b & e \\
c & f
\end{pmatrix}
\]

Transposing a matrix involves converting its rows into columns and vice versa. This operation is fundamental in linear algebra for various computations and transformations.
"""


# ==== Code ====
def transpose_matrix(a: list[list[int | float]]) -> list[list[int | float]]:
    # rows = len(a)
    cols = len(a[0])

    # initialize matrix
    b = []
    for _ in range(cols):
        b.append([])

    for i, _ in enumerate(a):
        for j, _ in enumerate(a[i]):
            b[j].append(a[i][j])

    return b


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"a": [[1, 2], [3, 4], [5, 6]]}
    print(inp)
    out = transpose_matrix(**inp)
    print(f"Output: {out}")
    exp = [[1, 3, 5], [2, 4, 6]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"a": [[1, 2, 3], [4, 5, 6]]}
    print(inp)
    out = transpose_matrix(**inp)
    print(f"Output: {out}")
    exp = [[1, 4], [2, 5], [3, 6]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
