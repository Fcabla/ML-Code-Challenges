"""
problem_id: 32
Category: machine learning
Difficulty: medium
URL: https://www.deep-ml.com/problem/Generate%20Polynomial%20Features
Page: 2

==== Title ====
Generate Polynomial Features

==== Description ====
Write a Python function to generate polynomial features for a given dataset. The function should take in a 2D numpy array X and an integer degree, and return a new 2D numpy array with polynomial features up to the specified degree.

==== Example ====
Example:
    X = np.array([[2, 3],
                  [3, 4],
                  [5, 6]])
    degree = 2
    output = polynomial_features(X, degree)
    print(output)
    # Output:
    # [[  1.   2.   3.   4.   6.   9.]
    #  [  1.   3.   4.   9.  12.  16.]
    #  [  1.   5.   6.  25.  30.  36.]]

    Reasoning:
    For each sample in X, the function generates all polynomial combinations
    of the features up to the given degree. For degree=2, it includes
    combinations like [x1^0, x1^1, x1^2, x2^0, x2^1, x2^2, x1^1*x2^1],
    where x1 and x2 are the features.

==== Learn More ====
Understanding Polynomial Features
Generating polynomial features is a method used to create new features for a machine learning model by raising existing features to a specified power. This technique helps in capturing non-linear relationships between features.
For instance, given a dataset with two features x1 and x2, generating polynomial features up to degree 2 will create new features such as x1^2, x2^2, and x1*x2. This expands the feature space, allowing a linear model to fit more complex, non-linear data.
In this problem, you will write a function to generate polynomial features for a given dataset. Given a 2D numpy array X and an integer degree, the function will create a new 2D numpy array with polynomial combinations of the features up to the specified degree.
This method is useful in algorithms like polynomial regression where capturing the relationship between features and the target variable requires polynomial terms.
By understanding and implementing this technique, you can enhance the performance of your models on datasets with non-linear relationships.
"""

from itertools import combinations_with_replacement

# ==== Code ====
import numpy as np


def polynomial_features(X, degree):
    n_samples, n_features = np.shape(X)

    # Generate all combinations of feature indices for polynomial terms
    # [(), (0,), (1,), (0, 0), (0, 1), (1, 1)]
    def index_combinations():
        combs = [
            combinations_with_replacement(range(n_features), i)
            for i in range(0, degree + 1)
        ]
        flat_combs = [item for sublist in combs for item in sublist]
        return flat_combs

    combinations = index_combinations()
    print(combinations)
    n_output_features = len(combinations)
    X_new = np.empty((n_samples, n_output_features))

    # Compute polynomial features
    for i, index_combs in enumerate(combinations):
        X_new[:, i] = np.prod(X[:, index_combs], axis=1)

    return X_new.tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"X": np.array([[2, 3], [3, 4], [5, 6]]), "degree": 2}
    print(inp)
    out = polynomial_features(**inp)
    print(f"Output: {out}")
    exp = [
        [1.0, 2.0, 3.0, 4.0, 6.0, 9.0],
        [1.0, 3.0, 4.0, 9.0, 12.0, 16.0],
        [1.0, 5.0, 6.0, 25.0, 30.0, 36.0],
    ]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"X": np.array([[1, 2], [3, 4], [5, 6]]), "degree": 3}
    print(inp)
    out = polynomial_features(**inp)
    print(f"Output: {out}")
    exp = [
        [1.0, 1.0, 2.0, 1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 8.0],
        [1.0, 3.0, 4.0, 9.0, 12.0, 16.0, 27.0, 36.0, 48.0, 64.0],
        [1.0, 5.0, 6.0, 25.0, 30.0, 36.0, 125.0, 150.0, 180.0, 216.0],
    ]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
