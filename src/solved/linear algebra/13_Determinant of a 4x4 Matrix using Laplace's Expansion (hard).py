"""
problem_id: 13
Category: linear algebra
Difficulty: hard
URL: https://www.deep-ml.com/problem/Determinant%20of%20a%204x4%20Matrix%20using%20Laplace's%20Expansion
Page: 1

==== Title ====
Determinant of a 4x4 Matrix using Laplace's Expansion (hard)

==== Description ====
Write a Python function that calculates the determinant of a 4x4 matrix using Laplace's Expansion method. The function should take a single argument, a 4x4 matrix represented as a list of lists, and return the determinant of the matrix. The elements of the matrix can be integers or floating-point numbers. Implement the function recursively to handle the computation of determinants for the 3x3 minor matrices.

==== Example ====
Example:
        input: a = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
        output: 0
        reasoning: Using Laplace's Expansion, the determinant of a 4x4 matrix is calculated by expanding it into minors and cofactors along any row or column. Given the symmetrical and linear nature of this specific matrix, its determinant is 0. The calculation for a generic 4x4 matrix involves more complex steps, breaking it down into the determinants of 3x3 matrices.

==== Learn More ====
Determinant of a 4x4 Matrix using Laplace's Expansion

Laplace's Expansion, also known as cofactor expansion, is a method to calculate the determinant of a square matrix of any size. For a 4x4 matrix \(A\), this method involves expanding \(A\) into minors and cofactors along a chosen row or column.

Consider a 4x4 matrix \(A\):
\[
A = \begin{pmatrix}
a_{11} & a_{12} & a_{13} & a_{14} \\
a_{21} & a_{22} & a_{23} & a_{24} \\
a_{31} & a_{32} & a_{33} & a_{34} \\
a_{41} & a_{42} & a_{43} & a_{44}
\end{pmatrix}
\]

The determinant of \(A\), \(\det(A)\), can be calculated by selecting any row or column (e.g., the first row) and using the formula that involves the elements of that row (or column), their corresponding cofactors, and the determinants of the 3x3 minor matrices obtained by removing the row and column of each element. This process is recursive, as calculating the determinants of the 3x3 matrices involves further expansions.

The expansion formula for the first row is as follows:
\[
\det(A) = a_{11}C_{11} - a_{12}C_{12} + a_{13}C_{13} - a_{14}C_{14}
\]

Here, \(C_{ij}\) represents the cofactor of element \(a_{ij}\), which is calculated as \((-1)^{i+j}\) times the determinant of the minor matrix obtained after removing the \(i\)th row and \(j\)th column from \(A\).
"""


# ==== Code ====
def determinant_4x4(matrix: list[list[int | float]]) -> float:
    # Your recursive implementation here
    # det(A) == a11*C11 - a12*C12 + a13*C13 - a14*C14
    # Cij = (-1)**(i+j) * determinant of the minor matrix after removing row i and col j

    # Recursive function Base case len = 1, det = the value
    if len(matrix) == 1:
        return matrix[0][0]
    # recursive case
    det = 0
    selected_row = 0
    for column in range(len(matrix)):
        # len(matrix) - 1 x len(matrix) - 1 shape removing ther ow and column of each elem
        minor = [row[:column] + row[column + 1 :] for row in matrix[selected_row + 1 :]]
        cofactor = ((-1) ** column) * determinant_4x4(minor)
        det += matrix[selected_row][column] * cofactor
    return det


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"matrix": [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]}
    print(inp)
    out = determinant_4x4(**inp)
    print(f"Output: {out}")
    exp = 0
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"matrix": [[4, 3, 2, 1], [3, 2, 1, 4], [2, 1, 4, 3], [1, 4, 3, 2]]}
    print(inp)
    out = determinant_4x4(**inp)
    print(f"Output: {out}")
    exp = -160
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {"matrix": [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6]]}
    print(inp)
    out = determinant_4x4(**inp)
    print(f"Output: {out}")
    exp = 0
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
