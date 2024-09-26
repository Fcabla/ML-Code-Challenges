"""
problem_id: 9
Category: linear algebra
Difficulty: medium
URL: https://www.deep-ml.com/problem/Matrix%20times%20Matrix
Page: 3

==== Title ====
Matrix times Matrix (medium)

==== Description ====
multiply two matrices together (return -1 if shapes of matrix dont aline), i.e. C = A dot product B

==== Example ====
Example:
        input: A = [[1,2],
                    [2,4]],
               B = [[2,1],
                    [3,4]]
        output:[[ 8,  9],
                [16, 18]]
        reasoning: 1*2 + 2*3 = 8;
                   2*2 + 3*4 = 16;
                   1*1 + 2*4 = 9;
                   2*1 + 4*4 = 18

Example 2:
        input: A = [[1,2],
                    [2,4]],
               B = [[2,1],
                    [3,4],
                    [4,5]]
        output: -1
        reasoning: the length of the rows of A does not equal
          the column length of B

==== Learn More ====
Matrix Multiplication

Consider two matrices \(A\) and \(B\), to demonstrate their multiplication, defined as follows:

- Matrix \(A\):
\[
A = \begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{pmatrix}
\]

- Matrix \(B\):
\[
B = \begin{pmatrix}
b_{11} & b_{12} \\
b_{21} & b_{22}
\end{pmatrix}
\]

The multiplication of matrix \(A\) by matrix \(B\) is calculated as:
\[
A \times B = \begin{pmatrix}
a_{11}b_{11} + a_{12}b_{21} & a_{11}b_{12} + a_{12}b_{22} \\
a_{21}b_{11} + a_{22}b_{21} & a_{21}b_{12} + a_{22}b_{22}
\end{pmatrix}
\]

This operation results in a new matrix where each element is the result of the dot product between the rows of matrix \(A\) and the columns of matrix \(B\).
"""


# ==== Code ====
def matrixmul(
    a: list[list[int | float]], b: list[list[int | float]]
) -> list[list[int | float]]:
    # a = nxm, b = mxp, c = nxp
    n, m = len(a), len(a[0])
    m_b, p = len(b), len(b[0])

    if m != m_b:
        return -1

    c = [[0 for _ in range(p)] for _ in range(n)]

    for i in range(n):
        for j in range(p):
            sum = 0
            for k in range(m):
                sum += a[i][k] * b[k][j]
            c[i][j] = sum
    return c


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "a": [[1, 2, 3], [2, 3, 4], [5, 6, 7]],
        "b": [[3, 2, 1], [4, 3, 2], [5, 4, 3]],
    }
    print(inp)
    out = matrixmul(**inp)
    print(f"Output: {out}")
    exp = [[26, 20, 14], [38, 29, 20], [74, 56, 38]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"a": [[0, 0], [2, 4], [1, 2]], "b": [[0, 0], [2, 4]]}
    print(inp)
    out = matrixmul(**inp)
    print(f"Output: {out}")
    exp = [[0, 0], [8, 16], [4, 8]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {"a": [[0, 0], [2, 4], [1, 2]], "b": [[0, 0, 1], [2, 4, 1], [1, 2, 3]]}
    print(inp)
    out = matrixmul(**inp)
    print(f"Output: {out}")
    exp = -1
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
