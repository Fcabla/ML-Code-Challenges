"""
problem_id: 1
Category: linear algebra
Difficulty: easy
URL: https://www.deep-ml.com/problem/Matrix%20times%20Vector
Page: 1

==== Title ====
Matrix times Vector (easy)

==== Description ====
Write a Python function that takes the dot product of a matrix and a vector. return -1 if the matrix could not be dotted with the vector

==== Example ====
Example:
        input: a = [[1,2],[2,4]], b = [1,2]
        output:[5, 10]
        reasoning: 1*1 + 2*2 = 5;
                   1*2+ 2*4 = 10

==== Learn More ====
Matrix Times Vector

Consider a matrix \(A\) and a vector \(v\), where:

Matrix \(A\):
\[
A = \begin{pmatrix}
a_{11} & a_{12} \\
a_{21} & a_{22}
\end{pmatrix}
\]

Vector \(v\):
\[
v = \begin{pmatrix}
v_1 \\
v_2
\end{pmatrix}
\]

The dot product of \(A\) and \(v\) results in a new vector:
\[
A \cdot v = \begin{pmatrix}
a_{11}v_1 + a_{12}v_2 \\
a_{21}v_1 + a_{22}v_2
\end{pmatrix}
\]

Things to note: an \(n \times m\) matrix will need to be multiplied by a vector of size \(m\) or else this will not work.
"""


# ==== Code ====
def matrix_dot_vector(
    a: list[list[int | float]], b: list[int | float]
) -> list[int | float]:
    if len(a) != len(b):
        return -1
    res = []
    for row in a:
        c = 0
        for a1, b1 in zip(row, b):
            c += a1 * b1
        res.append(c)
    return res


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"a": [[1, 2, 3], [2, 4, 5], [6, 8, 9]], "b": [1, 2, 3]}
    print(inp)
    out = matrix_dot_vector(**inp)
    print(f"Output: {out}")
    exp = [14, 25, 49]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"a": [[1, 2], [2, 4], [6, 8], [12, 4]], "b": [1, 2, 3]}
    print(inp)
    out = matrix_dot_vector(**inp)
    print(f"Output: {out}")
    exp = -1
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
