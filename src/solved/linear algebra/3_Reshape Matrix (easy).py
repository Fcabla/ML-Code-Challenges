"""
problem_id: 3
Category: linear algebra
Difficulty: easy
URL: https://www.deep-ml.com/problem/Reshape%20Matrix
Page: 2

==== Title ====
Reshape Matrix (easy)

==== Description ====
Write a Python function that reshapes a given matrix into a specified shape.

==== Example ====
Example:
        input: a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)
        output: [[1, 2], [3, 4], [5, 6], [7, 8]]
        reasoning: The given matrix is reshaped from 2x4 to 4x2.

==== Learn More ====
Reshaping a Matrix

Matrix reshaping involves changing the shape of a matrix without altering its data. This is essential in many machine learning tasks where the input data needs to be formatted in a specific way.

For example, consider a matrix \(M\):

Original Matrix \(M\):
\[
M = \begin{pmatrix}
1 & 2 & 3 & 4 \\
5 & 6 & 7 & 8
\end{pmatrix}
\]

Reshaped Matrix \(M'\) with shape (4, 2):
\[
M' = \begin{pmatrix}
1 & 2 \\
3 & 4 \\
5 & 6 \\
7 & 8
\end{pmatrix}
\]

Ensure the total number of elements remains constant during reshaping.
"""

# ==== Code ====


def reshape_matrix(
    a: list[list[int | float]], new_shape: tuple[int, int]
) -> list[list[int | float]]:
    # Write your code here and return a python list after reshaping by using numpy's tolist() method
    flatten = [item for row in a for item in row]
    reshaped_matrix = []
    for i in range(new_shape[1], len(flatten) + 1, new_shape[1]):
        reshaped_matrix.append(flatten[i - new_shape[1] : i])
    return reshaped_matrix


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"a": [[1, 2, 3, 4], [5, 6, 7, 8]], "new_shape": (4, 2)}
    print(inp)
    out = reshape_matrix(**inp)
    print(f"Output: {out}")
    exp = [[1, 2], [3, 4], [5, 6], [7, 8]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"a": [[1, 2, 3], [4, 5, 6]], "new_shape": (3, 2)}
    print(inp)
    out = reshape_matrix(**inp)
    print(f"Output: {out}")
    exp = [[1, 2], [3, 4], [5, 6]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {"a": [[1, 2, 3, 4], [5, 6, 7, 8]], "new_shape": (2, 4)}
    print(inp)
    out = reshape_matrix(**inp)
    print(f"Output: {out}")
    exp = [[1, 2, 3, 4], [5, 6, 7, 8]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
