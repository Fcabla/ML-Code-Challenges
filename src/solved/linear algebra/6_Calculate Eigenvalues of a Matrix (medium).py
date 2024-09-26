"""
problem_id: 6
Category: linear algebra
Difficulty: medium
URL: https://www.deep-ml.com/problem/Calculate%20Eigenvalues%20of%20a%20Matrix
Page: 3

==== Title ====
Calculate Eigenvalues of a Matrix (medium)

==== Description ====
Write a Python function that calculates the eigenvalues of a 2x2 matrix. The function should return a list containing the eigenvalues, sort values from highest to lowest.

==== Example ====
Example:
        input: matrix = [[2, 1], [1, 2]]
        output: [3.0, 1.0]
        reasoning: The eigenvalues of the matrix are calculated using the characteristic equation of the matrix, which for a 2x2 matrix is $\lambda^2 - 	ext{trace}(A)\lambda + 	ext{det}(A) = 0$, where $\lambda$ are the eigenvalues.

==== Learn More ====
Calculate Eigenvalues

Eigenvalues of a matrix offer significant insight into the matrix's behavior, particularly in the context of linear transformations and systems of linear equations.

 Definition
For a square matrix \(A\), eigenvalues are scalars \(\lambda\) that satisfy the equation for some non-zero vector \(v\) (eigenvector):
\[
Av = \lambda v
\]

 Calculation for a 2x2 Matrix
The eigenvalues of a 2x2 matrix \(A\), given by:
\[
A = \begin{pmatrix}
a & b \\
c & d
\end{pmatrix}
\]
are determined by solving the characteristic equation:
\[
\det(A - \lambda I) = 0
\]
This simplifies to a quadratic equation:
\[
\lambda^2 - \text{tr}(A) \lambda + \det(A) = 0
\]
Here, the trace of A, denoted as tr(A), is \(a + d\), and the determinant of A, denoted as det(A), is \(ad - bc\). Solving this equation yields the eigenvalues, \(\lambda\).

 Significance
Understanding eigenvalues is essential for analyzing the effects of linear transformations represented by the matrix. They are crucial in various applications, including stability analysis, vibration analysis, and Principal Component Analysis (PCA) in machine learning.
"""


# ==== Code ====
def calculate_eigenvalues(matrix: list[list[float | int]]) -> list[float]:
    # Calculate determinant #a*d - b*c
    det = (matrix[0][0] * matrix[1][1]) - (matrix[0][1] * matrix[1][0])

    # trace = diagonal
    tr = -sum([matrix[i][i] for i in range(len(matrix))])

    a = 1
    # quadratic a-tr+det=0
    # (-b +- sqrt(b** - 4*a*c))/2*a
    x1 = (-tr + (tr**2 - 4 * a * det) ** 0.5) / 2 * a
    x2 = (-tr - (tr**2 - 4 * a * det) ** 0.5) / 2 * a
    eigenvalues = [x1, x2]
    eigenvalues.sort(reverse=True)
    return eigenvalues


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"matrix": [[2, 1], [1, 2]]}
    print(inp)
    out = calculate_eigenvalues(**inp)
    print(f"Output: {out}")
    exp = [3.0, 1.0]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"matrix": [[4, -2], [1, 1]]}
    print(inp)
    out = calculate_eigenvalues(**inp)
    print(f"Output: {out}")
    exp = [3.0, 2.0]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
