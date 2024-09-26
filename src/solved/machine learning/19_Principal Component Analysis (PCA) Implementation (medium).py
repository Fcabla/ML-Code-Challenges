"""
problem_id: 19
Category: machine learning
Difficulty: medium
URL: https://www.deep-ml.com/problem/Principal%20Component%20Analysis%20(PCA)%20Implementation
Page: 3

==== Title ====
Principal Component Analysis (PCA) Implementation (medium)

==== Description ====
Write a Python function that performs Principal Component Analysis (PCA) from scratch. The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature. The function should standardize the dataset, compute the covariance matrix, find the eigenvalues and eigenvectors, and return the principal components (the eigenvectors corresponding to the largest eigenvalues). The function should also take an integer k as input, representing the number of principal components to return.

==== Example ====
Example:
        input: data = np.array([[1, 2], [3, 4], [5, 6]]), k = 1
        output:  [[0.7071], [0.7071]]
        reasoning: After standardizing the data and computing the covariance matrix, the eigenvalues and eigenvectors are calculated. The largest eigenvalue's corresponding eigenvector is returned as the principal component, rounded to four decimal places.

==== Learn More ====
Understanding Eigenvalues in PCA

Principal Component Analysis (PCA) utilizes the concept of eigenvalues and eigenvectors to identify the principal components of a dataset. Here's how eigenvalues fit into the PCA process:

 Eigenvalues and Eigenvectors: The Foundation of PCA

For a given square matrix \(A\), representing the covariance matrix in PCA, eigenvalues \(\lambda\) and their corresponding eigenvectors \(v\) satisfy:
\[
Av = \lambda v
\]

 Calculating Eigenvalues

The eigenvalues of matrix \(A\) are found by solving the characteristic equation:
\[
\det(A - \lambda I) = 0
\]
where \(I\) is the identity matrix of the same dimension as \(A\). This equation highlights the relationship between a matrix, its eigenvalues, and eigenvectors.

 Role in PCA

In PCA, the covariance matrix's eigenvalues represent the variance explained by its eigenvectors. Thus, selecting the eigenvectors associated with the largest eigenvalues is akin to choosing the principal components that retain the most data variance.

 Eigenvalues and Dimensionality Reduction

The magnitude of an eigenvalue correlates with the importance of its corresponding eigenvector (principal component) in representing the dataset's variability. By selecting a subset of eigenvectors corresponding to the largest eigenvalues, PCA achieves dimensionality reduction while preserving as much of the dataset's variability as possible.

 Practical Application

Standardize the Dataset: Ensure that each feature has a mean of 0 and a standard deviation of 1.
Compute the Covariance Matrix: Reflects how features vary together.
Find Eigenvalues and Eigenvectors: Solve the characteristic equation for the covariance matrix.
Select Principal Components: Choose eigenvectors (components) with the highest eigenvalues for dimensionality reduction.

Through this process, PCA transforms the original features into a new set of uncorrelated features (principal components), ordered by the amount of original variance they explain.
"""

# ==== Code ====
import numpy as np


def pca(data: np.ndarray, k: int) -> list[list[int | float]]:
    # standarize data
    data_standardized = (data - data.mean(axis=0)) / (data.std(axis=0))
    # compute covariance
    covariance_matrix = np.cov(data_standardized, rowvar=False)
    # Obtain eigens
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort the eigenvectors by decreasing eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    # eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:, idx]
    principal_components = eigenvectors_sorted[:, :k]
    # Your code here
    return np.round(principal_components, 4).tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"data": np.array([[4, 2, 1], [5, 6, 7], [9, 12, 1], [4, 6, 7]]), "k": 2}
    print(inp)
    out = pca(**inp)
    print(f"Output: {out}")
    exp = [[0.6855, 0.0776], [0.6202, 0.4586], [-0.3814, 0.8853]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"data": np.array([[1, 2], [3, 4], [5, 6]]), "k": 1}
    print(inp)
    out = pca(**inp)
    print(f"Output: {out}")
    exp = [[0.7071], [0.7071]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
