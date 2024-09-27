"""
problem_id: 34
Category: machine learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/One-Hot%20Encoding%20of%20Nominal%20Values
Page: 2

==== Title ====
One-Hot Encoding of Nominal Values

==== Description ====
Write a Python function to perform one-hot encoding of nominal values. The function should take in a 1D numpy array x of integer values and an optional integer n_col representing the number of columns for the one-hot encoded array. If n_col is not provided, it should be automatically determined from the input array.

==== Example ====
Example:
    x = np.array([0, 1, 2, 1, 0])
    output = to_categorical(x)
    print(output)
    # Output:
    # [[1. 0. 0.]
    #  [0. 1. 0.]
    #  [0. 0. 1.]
    #  [0. 1. 0.]
    #  [1. 0. 0.]]

    Reasoning:
    Each element in the input array is transformed into a one-hot encoded vector,
    where the index corresponding to the value in the input array is set to 1,
    and all other indices are set to 0.

==== Learn More ====
Understanding One-Hot Encoding
One-hot encoding is a method used to represent categorical variables as binary vectors. This technique is useful in machine learning when dealing with categorical data that has no ordinal relationship.
In one-hot encoding, each category is represented by a binary vector with a length equal to the number of categories. The vector has a value of 1 at the index corresponding to the category and 0 at all other indices.
For example, if you have three categories: 0, 1, and 2, the one-hot encoded vectors would be:

0: \( \left[1, 0, 0\right] \)
1: \( \left[0, 1, 0\right] \)
2: \( \left[0, 0, 1\right] \)

This method ensures that the model does not assume any ordinal relationship between categories, which is crucial for many machine learning algorithms. The one-hot encoding process can be mathematically represented as follows:
Given a category \( x_i \) from a set of categories \( \{0, 1, \ldots, n-1\} \), the one-hot encoded vector \( \mathbf{v} \) is:

\[ \mathbf{v}_i =
\begin{cases}
1 & \text{if } i = x_i \\
0 & \text{otherwise}
\end{cases}
\]

This vector \( \mathbf{v} \) will have a length equal to the number of unique categories.
"""

# ==== Code ====
import numpy as np


def to_categorical(x, n_col=None):
    # Your code here
    # Get unique items
    categories = list(set(x))
    categories.sort()

    if not n_col:
        n_col = len(categories)

    # Create empty matrix with zeros
    result = np.zeros((x.shape[0], n_col))
    result[np.arange(x.shape[0]), x] = 1

    # Other solution
    # result = []
    # for el in x:
    # ohe = np.zeros(n_col)
    # if not n_col:
    # 	ohe[categories.index(el)] = 1
    # else:
    # 	ohe[categories.index(el)+1] = 1
    # result.append(ohe.tolist())
    return result.tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"x": np.array([0, 1, 2, 1, 0]), "n_col": None}
    print(inp)
    out = to_categorical(**inp)
    print(f"Output: {out}")
    exp = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
    ]

    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"x": np.array([3, 1, 2, 1, 3]), "n_col": 4}
    print(inp)
    out = to_categorical(**inp)
    print(f"Output: {out}")
    exp = [
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
    # Test case 3
