"""
problem_id: 48
Category: linear algebra
Difficulty: medium
URL: https://www.deep-ml.com/problem/Reduced%20Row%20Echelon%20Form
Page: 3

==== Title ====
Implement Reduced Row Echelon Form (RREF) Function

==== Description ====
In this problem, your task is to implement a function that converts a given matrix into its Reduced Row Echelon Form (RREF). The RREF of a matrix is a special form where each leading entry in a row is 1, and all other elements in the column containing the leading 1 are zeros, except for the leading 1 itself.
However, there are some additional details to keep in mind:

Diagonal entries can be 0 if the matrix is reducible (i.e., the row corresponding to that position can be eliminated entirely).
Some rows may consist entirely of zeros.
If a column contains a pivot (a leading 1), all other entries in that column should be zero.

Your task is to implement the RREF algorithm, which must handle these cases, and convert any given matrix into its RREF.

==== Example ====
Example:
import numpy as np

matrix = np.array([
    [1, 2, -1, -4],
    [2, 3, -1, -11],
    [-2, 0, -3, 22]
])

rref_matrix = rref(matrix)
print(rref_matrix)

# Expected Output:
# array([
#    [ 1.  0.  0. -8.],
#    [ 0.  1.  0.  1.],
#    [-0. -0.  1. -2.]
# ])

==== Learn More ====
Understanding the RREF Algorithm
The Reduced Row Echelon Form (RREF) of a matrix is a specific form of a matrix achieved through a sequence of elementary row operations. This algorithm will convert any matrix into its RREF, which is useful in solving linear equations and understanding the properties of the matrix.
Here’s a step-by-step guide to implementing the RREF algorithm:
1. Start with the leftmost column
Set the initial leading column to the first column of the matrix. We'll move this "lead" to the right as we progress through the algorithm.
2. Select the pivot row
Identify the first non-zero entry in the current leading column. This entry is known as the pivot. If necessary, add the row containing the pivot to the current row to avoid it being zero.
3. Scale the pivot row
Divide the entire pivot row by the pivot value to make the leading entry equal to 1.

\[
\text{Row}_r = \frac{\text{Row}_r}{\text{pivot}}
\]

For example, if the pivot is 3, then divide the entire row by 3 to make the leading entry 1.
4. Eliminate above and below the pivot
Subtract multiples of the pivot row from all the other rows to create zeros in the rest of the pivot column. This ensures that the pivot is the only non-zero entry in its column.

\[
\text{Row}_i = \text{Row}_i - (\text{Row}_r \times \text{lead coefficient})
\]

Repeat this step for each row \( i \) where \( i \neq r \), ensuring that all entries above and below the pivot are zero.
5. Move to the next column
Move the lead one column to the right and repeat the process from step 2. Continue until there are no more columns to process or the remaining submatrix is all zeros.
By following these steps, the matrix will be converted into its Reduced Row Echelon Form, where each leading entry is 1, and all other entries in the leading columns are zero.
"""

# ==== Code ====
import numpy as np


def rref(matrix):
    # Your code here
    n, m = matrix.shape
    matrix = matrix.astype(np.float32)
    # row-echelon form
    for i in range(n):
        # Find the maximum row to swap with if needed
        max_row = i + np.argmax(np.abs(matrix[i:, i]))

        # Swap if the pivot is not already the maximum
        if max_row != i:
            matrix[[i, max_row]] = matrix[[max_row, i]]

        if matrix[i, i] != 0:
            for j in range(i + 1, n):
                # Avoid division by zero
                if matrix[j, i] != 0:
                    matrix[j, :] -= matrix[i, :] * (matrix[j, i] / matrix[i, i])

    # move any row thats all 0s to the bottom of the matrix
    non_zero_rows = matrix[~np.all(matrix == 0, axis=1)]
    zero_rows = matrix[np.all(matrix == 0, axis=1)]
    # Concatenate non-zero rows with zero rows at the bottom
    result = np.vstack((non_zero_rows, zero_rows))

    # reduced row-echelon form
    # Make the pivots == 1
    for i in range(n):
        # Ensure the pivot is 1
        if result[i, i] != 0:
            result[i, :] /= result[i, i]
        # Eliminate non-zero entries in the current column above the pivot
        for j in range(i):
            if result[j, i] != 0:
                result[j, :] -= result[j, i] * result[i, :]

    return result.tolist()


""" Solution:
def rref(matrix):
    # Convert to float for division operations
    A = matrix.astype(np.float32)
    n, m = A.shape

    for i in range(n):
        if A[i, i] == 0:
            nonzero_rel_id = np.nonzero(A[i:, i])[0]
            if len(nonzero_rel_id) == 0: continue

            A[i] = A[i] + A[nonzero_rel_id[0] + i]

        A[i] = A[i] / A[i, i]
        for j in range(n):
            if i != j:
                A[j] -= A[j, i] * A[i]

    return A
"""

# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"matrix": np.array([[1, 2, -1, -4], [2, 3, -1, -11], [-2, 0, -3, 22]])}
    print(inp)
    out = rref(**inp)
    print(f"Output: {out}")
    exp = [[1.0, 0.0, 0.0, -8.0], [0.0, 1.0, 0.0, 1.0], [-0.0, -0.0, 1.0, -2.0]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"matrix": np.array([[2, 4, -2], [4, 9, -3], [-2, -3, 7]])}
    print(inp)
    out = rref(**inp)
    print(f"Output: {out}")
    exp = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {"matrix": np.array([[0, 2, -1, -4], [2, 0, -1, -11], [-2, 0, 0, 22]])}
    print(inp)
    out = rref(**inp)
    print(f"Output: {out}")
    exp = [[1.0, 0.0, 0.0, -11.0], [-0.0, 1.0, 0.0, -7.5], [-0.0, -0.0, 1.0, -11.0]]

    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 4
    print("Test Case 4:")
    inp = {"matrix": np.array([[1, 2, -1], [2, 4, -1], [-2, -4, -3]])}
    print(inp)
    out = rref(**inp)
    print(f"Output: {out}")
    exp = [[1.0, 2.0, 0.0], [0.0, 0.0, 0.0], [-0.0, -0.0, 1.0]]

    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
