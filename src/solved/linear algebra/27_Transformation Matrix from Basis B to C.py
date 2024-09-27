"""
problem_id: 27
Category: linear algebra
Difficulty: easy
URL: https://www.deep-ml.com/problem/Transformation%20Matrix%20from%20Basis%20B%20to%20C
Page: 1

==== Title ====
Transformation Matrix from Basis B to C

==== Description ====
Given basis vectors in two different bases B and C for R^3, write a Python function to compute the transformation matrix P from basis B to C.

==== Example ====
Example:
        B = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]
        C = [[1, 2.3, 3],
             [4.4, 25, 6],
             [7.4, 8, 9]]
        output: [[-0.6772, -0.0126, 0.2342],
                [-0.0184, 0.0505, -0.0275],
                [0.5732, -0.0345, -0.0569]]

        reasoning: The transformation matrix P from basis B to C can be found using matrix operations involving the inverse of matrix C.

==== Learn More ====
Understanding Transformation Matrices
A transformation matrix allows us to convert the coordinates of a vector in one basis to coordinates in another basis. For bases B and C of a vector space, the transformation matrix P from B to C is calculated by:

Inverse of Basis C: First, find the inverse of the matrix representing basis C, denoted \(C^{-1}\).
Matrix Multiplication: Multiply \(C^{-1}\) by the matrix of basis B. The result is the transformation matrix \(P\), where \(P = C^{-1} \cdot B\).

This matrix \(P\) can be used to transform any vector coordinates from the B basis to the C basis.
Resources: Change of basis | Chapter 13, Essence of linear algebra by 3Blue1Brown
"""

# ==== Code ====
import numpy as np


def transform_basis(B: list[list[int]], C: list[list[int]]) -> list[list[float]]:
    # P = C^-1*B
    B = np.array(B)
    C = np.array(C)

    C_inv = np.linalg.inv(C)
    P = np.round(C_inv @ B, 4)

    return P.tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "B": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "C": [[1, 2.3, 3], [4.4, 25, 6], [7.4, 8, 9]],
    }
    print(inp)
    out = transform_basis(**inp)
    print(f"Output: {out}")
    exp = [
        [-0.6772, -0.0126, 0.2342],
        [-0.0184, 0.0505, -0.0275],
        [0.5732, -0.0345, -0.0569],
    ]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"B": [[1, 0], [0, 1]], "C": [[1, 2], [9, 2]]}
    print(inp)
    out = transform_basis(**inp)
    print(f"Output: {out}")
    exp = [[-0.125, 0.125], [0.5625, -0.0625]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
