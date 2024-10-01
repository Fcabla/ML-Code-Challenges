"""
problem_id: 12
Category: linear algebra
Difficulty: hard
URL: https://www.deep-ml.com/problem/Singular%20Value%20Decomposition%20(SVD)
Page: 1

==== Title ====
Singular Value Decomposition (SVD) (hard)

==== Description ====
Write a Python function that approximates the Singular Value Decomposition on a 2x2 matrix by using the jacobian method and without using numpy svd function, i mean you could but you wouldn't learn anything. return the result in this format.

==== Example ====
Example:
        input: a = [[2, 1], [1, 2]]
        output: (array([[-0.70710678, -0.70710678],
                        [-0.70710678,  0.70710678]]),
        array([3., 1.]),
        array([[-0.70710678, -0.70710678],
               [-0.70710678,  0.70710678]]))
        reasoning: U is the first matrix sigma is the second vector and V is the third matrix

==== Learn More ====
Singular Value Decomposition (SVD) via the Jacobi Method
Singular Value Decomposition (SVD) is a powerful matrix decomposition technique in linear algebra that expresses a matrix as the product of three other matrices, revealing its intrinsic geometric and algebraic properties. When using the Jacobi method, SVD decomposes a matrix \(A\) into:

\[
A = U\Sigma V^T
\]


\(A\) is the original \(m \times n\) matrix.
\(U\) is an \(m \times m\) orthogonal matrix whose columns are the left singular vectors of \(A\).
\(\Sigma\) is an \(m \times n\) diagonal matrix containing the singular values of \(A\).
\(V^T\) is the transpose of an \(n \times n\) orthogonal matrix whose columns are the right singular vectors of \(A\).

The Jacobi Method for SVD
The Jacobi method is an iterative algorithm used for diagonalizing a symmetric matrix through a series of rotational transformations. It is particularly suited for computing the SVD by iteratively applying rotations to minimize off-diagonal elements until the matrix is diagonal.
Steps of the Jacobi SVD Algorithm

Initialization: Start with \(A^TA\) (or \(AA^T\) for \(U\)) and set \(V\) (or \(U\)) as an identity matrix. The goal is to diagonalize \(A^TA\), obtaining \(V\) in the process.
Choosing Rotation Targets: Identify off-diagonal elements in \(A^TA\) to be minimized or zeroed out through rotations.
Calculating Rotation Angles: For each target off-diagonal element, calculate the angle \(\theta\) for the Jacobi rotation matrix \(J\) that would zero it. This involves solving for \(\theta\) using \(\text{atan2}\) to accurately handle the quadrant of rotation:
    \[
    \theta = 0.5 \cdot \text{atan2}(2a_{ij}, a_{ii} - a_{jj})
    \]
    where \(a_{ij}\) is the target off-diagonal element, and \(a_{ii}\), \(a_{jj}\) are the diagonal elements of \(A^TA\).

Applying Rotations: Construct \(J\) using \(\theta\) and apply the rotation to \(A^TA\), effectively reducing the magnitude of the target off-diagonal element. Update \(V\) (or \(U\)) by multiplying it by \(J\).
Iteration and Convergence: Repeat the process of selecting off-diagonal elements, calculating rotation angles, and applying rotations until \(A^TA\) is sufficiently diagonalized.
Extracting SVD Components: Once diagonalized, the diagonal entries of \(A^TA\) represent the squared singular values of \(A\). The matrices \(U\) and \(V\) are constructed from the accumulated rotations, containing the left and right singular vectors of \(A\), respectively.

Practical Considerations

The Jacobi method is particularly effective for dense matrices where off-diagonal elements are significant.
Careful implementation is required to ensure numerical stability and efficiency, especially for large matrices.
The iterative nature of the Jacobi method makes it computationally intensive, but it is highly parallelizable.
"""

# ==== Code ====
import numpy as np


def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    A_T_A = A.T @ A
    theta = 0.5 * np.arctan2(2 * A_T_A[0, 1], A_T_A[0, 0] - A_T_A[1, 1])
    j = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    A_prime = j.T @ A_T_A @ j

    # Calculate singular values from the diagonalized A^TA (approximation for 2x2 case)
    singular_values = np.sqrt(np.diag(A_prime))

    # Process for AA^T, if needed, similar to A^TA can be added here for completeness

    return (
        np.round(j.T, 8).tolist(),
        np.round(singular_values, 8).tolist(),
        np.round(j, 8).tolist(),
    )


# ==== Test cases ====

if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"A": np.array([[2, 1], [1, 2]])}
    print(inp)
    out = svd_2x2_singular_values(**inp)
    print(f"Output: {out}")
    exp = (
        [[0.70710678, -0.70710678], [0.70710678, 0.70710678]],
        [3.0, 1.0],
        [[0.70710678, 0.70710678], [-0.70710678, 0.70710678]],
    )
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"A": np.array([[1, 2], [3, 4]])}
    print(inp)
    out = svd_2x2_singular_values(**inp)
    print(f"Output: {out}")
    exp = (
        [[0.57604844, -0.81741556], [0.81741556, 0.57604844]],
        [5.4649857, 0.36596619],
        [[0.57604844, 0.81741556], [-0.81741556, 0.57604844]],
    )
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

# My solution v2
"""
def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    A = np.array(A)
    m, n = A.shape

    # A = U*E*V_t
    # Initialize variables
    U = np.identity(m)
    V = np.identity(n)
    E = np.zeros((m, n))
    AT_A = A.T @ A
    # Copy AT_A to work on the rotations
    A = AT_A.copy()

    # Calculate the rotation angle θ to annihilate the off-diagonal elements
    theta = 0.5 * np.arctan2(2 * A[0][1], (A[0][0] - A[1][1]))
    # Construct the Jacobi rotation matrix
    jac = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    # Apply the rotation matrix (mult)
    A_p = jac.T @ A @ jac  # (rotation from both sides)
    # Uptade U and V
    U = U @ jac  # (left singular vectors)
    V = V @ jac  # (right singular vectors)

    # singular_values = np.diag(A)
    singular_values = np.sqrt(np.diag(A_p))
    return (
        np.round(jac.T, 8).tolist(),
        np.round(singular_values, 8).tolist(),
        np.round(jac, 8).tolist(),
    )
    """
# My initial solution
"""
for i in range(m):
        for j in range(n):
            # For each pair of off-diagonal elements in A where i ≠ j
            # In this case not necessary (2x2 matrix points: (0,1), (1,0))
            if i != j:
                # Calculate the rotation angle θ to annihilate the off-diagonal elements
                theta = 0.5 * np.arctan2(2 * A[i][j], (A[i][i] - A[j][j]))

                # Construct the Jacobi rotation matrix
                jac = np.zeros((2, 2))
                jac[i][i] = np.cos(theta)
                jac[j][j] = np.cos(theta)
                jac[i][j] = -np.sin(theta)
                jac[j][i] = np.sin(theta)

                # Apply the rotation matrix (mult)
                A_p = jac.T @ A @ jac  # (rotation from both sides)

                # Uptade U and V
                U = U @ jac  # (left singular vectors)
                V = V @ jac  # (right singular vectors)

                # singular_values = np.diag(A)
                singular_values = np.sqrt(np.diag(A_p))
    return (
        np.round(jac.T, 8).tolist(),
        np.round(singular_values, 8).tolist(),
        np.round(jac, 8).tolist(),
    )

"""
