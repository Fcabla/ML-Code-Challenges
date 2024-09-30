"""
problem_id: 55
Category: linear algebra
Difficulty: medium
URL: https://www.deep-ml.com/problem/2D%20Translation%20Matrix
Page: 3

==== Title ====
2D Translation Matrix Implementation

==== Description ====
Task: Implement a 2D Translation Matrix
Your task is to implement a function that applies a 2D translation matrix to a set of points. A translation matrix is used to move points in 2D space by a specified distance in the x and y directions.
Write a function translate_object(points, tx, ty) where points is a list of [x, y] coordinates and tx and ty are the translation distances in the x and y directions respectively.
The function should return a new list of points after applying the translation matrix.

==== Example ====
Example:
import numpy as np

points = [[0, 0], [1, 0], [0.5, 1]]
tx, ty = 2, 3

print(translate_object(points, tx, ty))

# Expected Output:
# [[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]

==== Learn More ====
2D Translation Matrix Implementation
The translation matrix is a fundamental concept in linear algebra and computer graphics, used to move points or objects in a 2D space.

For a 2D translation, we use a 3x3 matrix to move a point (x, y) by x_t units in the x-direction and y_t units in the y-direction.

For any point P in 2D Cartesian space with coordinates (x, y), we can represent it in homogeneous coordinates as (x, y, 1):

\[
P_{\text{Cartesian}} = (x, y) \rightarrow P_{\text{Homogeneous}} = (x, y, 1)
\]


More generally, any scalar multiple of (x, y, 1) represents the same point in 2D space. So (kx, ky, k) for any non-zero k also represents the same point (x, y) in 2D space.
The addition of this third coordinate allows us to represent translation as a linear transformation.

The translation matrix is defined as:

\[
T = \begin{bmatrix}
1 & 0 & x_t \\
0 & 1 & y_t \\
0 & 0 & 1
\end{bmatrix}
\]

To apply this translation to a point (x, y) we use homogeneous coordinates by representing the point as (x, y, 1). Then the transformation is performed by matrix multiplication:

\[
\begin{bmatrix}
1 & 0 & x_t \\
0 & 1 & y_t \\
0 & 0 & 1
\end{bmatrix}
\begin{bmatrix}
x \\
y \\
1
\end{bmatrix}
=
\begin{bmatrix}
x + x_t \\
y + y_t \\
1
\end{bmatrix}
\]

Where:

(x, y) is the original point,
x_t is the translation in the x-direction,
y_t is the translation in the y-direction,
(x + x_t, y + y_t) is the resulting translated point.
"""

# ==== Code ====


def translate_object(points, tx, ty):
    for i in range(len(points)):
        points[i][0] += tx
        points[i][1] += ty
    translated_points = points
    """
    translation_matrix = np.array([
        [1, 0, tx],
        [0, 1, ty],
        [0, 0, 1]
    ])

    homogeneous_points = np.hstack([np.array(points), np.ones((len(points), 1))])

    translated_points = np.dot(homogeneous_points, translation_matrix.T)

    return translated_points[:, :2].tolist()
    """
    return translated_points


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"points": [[0, 0], [1, 0], [0.5, 1]], "tx": 2, "ty": 3}
    print(inp)
    out = translate_object(**inp)
    print(f"Output: {out}")
    exp = [[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"points": [[0, 0], [1, 0], [1, 1], [0, 1]], "tx": -1, "ty": 2}
    print(inp)
    out = translate_object(**inp)
    print(f"Output: {out}")
    exp = [[-1.0, 2.0], [0.0, 2.0], [0.0, 3.0], [-1.0, 3.0]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
