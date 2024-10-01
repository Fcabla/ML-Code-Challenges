"""
problem_id: 28
Category: linear algebra
Difficulty: hard
URL: https://www.deep-ml.com/problem/SVD%20of%20a%202x2%20Matrix%20using%20eigen%20values%20&%20vectors
Page: 2

==== Title ====
SVD of a 2x2 Matrix using eigen values & vectors

==== Description ====
Given a 2x2 matrix, write a Python function to compute its Singular Value Decomposition (SVD). The function should return the matrices U, S, and V such that A = U * S * V, use the method described in this post https://metamerist.blogspot.com/2006/10/linear-algebra-for-graphics-geeks-svd.html

==== Example ====
Example:
    A = [[-10, 8],
         [10, -1]]
    output:(array([[  0.8, -0.6], [-0.6, -0.8]]),
    array([15.65247584,  4.47213595]),
    array([[ -0.89442719,  0.4472136], [ -0.4472136 , -0.89442719]]))

==== Learn More ====
Understanding Singular Value Decomposition (SVD)
Singular Value Decomposition (SVD) is a method in linear algebra for decomposing a matrix into three other matrices. For a given matrix A, SVD is represented as:
\(A = U \cdot S \cdot V^T\)
Here's a step-by-step method to calculate the SVD of a 2x2 matrix by hand:

Calculate \(A^T A\) and \(A A^T\): Compute the product of the matrix with its transpose and the transpose of the matrix with itself. These matrices share the same eigenvalues.
Find the Eigenvalues: To find the eigenvalues of a 2x2 matrix, solve the characteristic equation \( \det(A - \lambda I) = 0 \). This results in a quadratic equation.
Compute the Singular Values: The singular values, which form the diagonal elements of the matrix S, are the square roots of the eigenvalues.
Calculate the Eigenvectors: For each eigenvalue, solve the equation \((A - \lambda I) \mathbf{x} = 0\) to find the corresponding eigenvector. Normalize these eigenvectors to form the columns of U and V.
Form the Matrices U, S, and V: Combine the singular values and eigenvectors to construct the matrices U, S, and V such that \(A = U \cdot S \cdot V^T\).

This method involves solving quadratic equations to find eigenvalues and eigenvectors and normalizing these vectors to unit length.
Resources: Linear Algebra for Graphics Geeks (SVD-IX) by METAMERIST
Robust algorithm for 2×2 SVD
"""

# ==== Code ====
import numpy as np


def svd_2x2(A: np.ndarray) -> tuple:
    # components used in the characteristic polynomial for eigenvalue calculations.
    y1, x1 = (A[1, 0] + A[0, 1]), (A[0, 0] - A[1, 1])
    y2, x2 = (A[1, 0] - A[0, 1]), (A[0, 0] + A[1, 1])

    # Calculate Hypotenuses:
    # magnitudes of the transformations represented by the matrix. They reflect how far the transformed vectors extend.
    h1 = np.sqrt(y1**2 + x1**2)
    h2 = np.sqrt(y2**2 + x2**2)

    # Normalization
    t1 = x1 / h1
    t2 = x2 / h2

    # Calculate Trigonometric Functions:
    # angles and rotations needed to form the orthogonal matrices U and V
    cc = np.sqrt((1.0 + t1) * (1.0 + t2))
    ss = np.sqrt((1.0 - t1) * (1.0 - t2))
    cs = np.sqrt((1.0 + t1) * (1.0 - t2))
    sc = np.sqrt((1.0 - t1) * (1.0 + t2))

    # Construct Matrix U
    # Eigenvector Formation: The components c1 and s1are derived from the previously calculated cc and ss
    # The orthogonal matrix U is constructed using these components.
    # Understanding U: consists of the normalized eigenvectors corresponding to the eigenvalues of AAT.
    # In SVD, U contains the left singular vectors.
    c1, s1 = (cc - ss) / 2.0, (sc + cs) / 2.0
    U = np.array([[-c1, -s1], [-s1, c1]])

    # Calculate singular values
    # Derived from the hypotenuses h1 and h2. They reflect the scaling factors of the transformations of A
    # This step gives the singular values, which correspond to the diagonal entries in matrix S. These values represent the "strength" of each transformation.
    s = np.array([(h1 + h2) / 2.0, abs(h1 - h2) / 2.0])

    # Matrix V is constructed by taking the inverse of the singular values and multiplying by the transpose of U and A.
    # This effectively uses the relationships between A, U, and S to find V.
    # Orthogonality: Since U is orthogonal, its transpose can be used to simplify the calculation. Matrix V contains the right singular vectors, which are also orthogonal.
    V = np.diag(1.0 / s) @ U.T @ A

    return U, s, V


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"A": np.array([[-10, 8], [10, -1]])}
    print(inp)
    U, s, V = svd_2x2(**inp)
    print(f"Output: {U,s,V}")
    out = np.round((U @ np.diag(s) @ V)).tolist()
    print(f"Output: {out}")
    exp = [[-10, 8], [10, -1]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"A": np.array([[1, 2], [3, 4]])}
    print(inp)
    U, s, V = svd_2x2(**inp)
    print(f"Output: {U,s,V}")
    out = np.round((U @ np.diag(s) @ V)).tolist()
    print(f"Output: {out}")
    exp = [[1, 2], [3, 4]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3

# Not working


def svd_2x2v1(A: np.ndarray) -> tuple:
    # Your code here
    # A = U * S * V

    # Calculate A@At
    AAt = A @ A.T  # AtA = A.T@A # They share eigens

    # Set Up the Characteristic Polynomial
    # det(AA−λI)=0
    a = AAt[0, 0]
    b = AAt[0, 1]
    c = AAt[1, 0]
    d = AAt[1, 1]

    # Calculate the Determinant
    # det([a, b],[c, d]) = ad−bc
    # det([a−λ, b],[c, d−λ]) = ad−bc
    # det(AA−λI)=(a−λ)(d−λ)−(−b)(−c)
    # (a−λ)(d−λ) = a⋅b−aλ−bλ+λ**2
    # (a−λ)(d−λ) = λ**2 - (a+b)λ + a⋅b
    # λ**2−(a+b)λ+(a*d−b*c)=0
    # coeffs = [1, -trace, det]
    coeffs = [1, -(a + d), (a * d - b * c)]

    # Solve for eigenvalues
    # eigenvalues = np.roots(coeffs)
    # Calculate eigenvalues using the quadratic formula
    eigenvalue1 = (coeffs[1] + np.sqrt(coeffs[1] ** 2 - 4 * coeffs[2])) / 2
    eigenvalue2 = (coeffs[1] - np.sqrt(coeffs[1] ** 2 - 4 * coeffs[2])) / 2

    # Step 5: Calculate singular values
    singular_values = np.sqrt(np.abs([eigenvalue1, eigenvalue2]))

    # Calculate eigenvectors for each eigenvalue
    # Calculate the Eigenvectors: For each eigenvalue, solve the equation to find the corresponding eigenvector. Normalize these eigenvectors to form the columns of U and V.
    U = []
    for eigenvalue in (eigenvalue1, eigenvalue2):
        # Solve (AAt - λI)X = 0
        # Construct the matrix (AAt - eigenvalue * I)
        matrix = AAt - eigenvalue * np.eye(2)
        # Find eigenvector by solving the linear system
        # The last row can be used if rank is reduced
        if matrix[0, 1] != 0:  # If off-diagonal is not zero
            eigenvector = np.array(
                [-matrix[1, 0], matrix[0, 0] - eigenvalue]
            )  # e.g., [-c, a - λ]
        else:
            # If off-diagonal is zero, just take a direct approach
            eigenvector = np.array([1, 0]) if matrix[0, 0] != 0 else np.array([0, 1])

        # Normalize the eigenvector
        eigenvector /= np.linalg.norm(eigenvector)
        U.append(eigenvector)

    U = np.array(U).T  # Convert to column vectors
    # Step 6: Form the Matrices U, S, and V: Combine the singular values and eigenvectors to construct the matrices U, S, and V such that
    S = np.zeros((2, 2))
    S[0, 0] = singular_values[0]
    S[1, 1] = singular_values[1]

    # Calculate V: Solve for eigenvectors of AtA
    AtA = A.T @ A
    V = []
    for eigenvalue in (eigenvalue1, eigenvalue2):
        # Solve (AtA - λI)Y = 0
        matrix = AtA - eigenvalue * np.eye(2)

        # Find eigenvector by solving the linear system
        if matrix[0, 1] != 0:
            eigenvector = np.array([-matrix[1, 0], matrix[0, 0] - eigenvalue])
        else:
            eigenvector = np.array([1, 0]) if matrix[0, 0] != 0 else np.array([0, 1])

        # Normalize the eigenvector
        eigenvector /= np.linalg.norm(eigenvector)
        V.append(eigenvector)

    V = np.array(V).T  # Convert to column vectors

    # Step 7: Return U, S, and V
    print(A == U @ S @ V.T)
    return U, S, V


def svd_2x2_v2(A):
    # Step 1: Calculate A^T * A and A * A^T
    A_T = A.T
    A_T_A = np.dot(A_T, A)
    A_A_T = np.dot(A, A_T)

    # Step 2: Find eigenvalues of A^T * A and A * A^T
    eigenvalues_A_T_A, _ = np.linalg.eig(A_T_A)
    eigenvalues_A_A_T, _ = np.linalg.eig(A_A_T)

    # Step 3: Compute singular values (square roots of eigenvalues)
    singular_values = np.sqrt(eigenvalues_A_T_A)

    # Step 4: Calculate eigenvectors of A^T * A (for V) and A * A^T (for U)
    _, eigenvectors_A_T_A = np.linalg.eig(A_T_A)
    _, eigenvectors_A_A_T = np.linalg.eig(A_A_T)

    # Normalize eigenvectors to form U and V
    U = eigenvectors_A_A_T / np.linalg.norm(eigenvectors_A_A_T, axis=0)
    V = eigenvectors_A_T_A / np.linalg.norm(eigenvectors_A_T_A, axis=0)

    # Step 5: Form the matrix S
    S = np.zeros_like(A, dtype=float)
    S[np.diag_indices(len(singular_values))] = singular_values
    print(U, S, V.T)
    return U, S, V.T
