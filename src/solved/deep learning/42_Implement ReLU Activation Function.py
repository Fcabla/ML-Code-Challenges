"""
problem_id: 42
Category: deep learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/ReLU%20Activation%20Function
Page: 2

==== Title ====
Implement ReLU Activation Function

==== Description ====
Write a Python function `relu` that implements the Rectified Linear Unit (ReLU) activation function. The function should take a single float as input and return the value after applying the ReLU function. The ReLU function returns the input if it's greater than 0, otherwise, it returns 0.

==== Example ====
Example:
print(relu(0))
# Output: 0

print(relu(1))
# Output: 1

print(relu(-1))
# Output: 0

==== Learn More ====
Understanding the RELU Activation Function
The ReLU (Rectified Linear Unit) activation function is widely used in neural networks, particularly in hidden layers of deep learning models. It maps any real-valued number to the non-negative range \([0, \infty)\], which helps introduce non-linearity into the model while maintaining computational efficiency.
Mathematical Definition
The ReLU function is mathematically defined as:

\[
f(z) = \max(0, z)
\]

Where \(z\) is the input to the function.
Characteristics

Output Range: The output is always in the range \([0, \infty)\). Values below 0 are mapped to 0, while positive values are retained.
Shape: The function has an "L" shaped curve with a horizontal axis at \(y = 0\) and a linear increase for positive \(z\).
Gradient: The gradient is 1 for positive values of \(z\) and 0 for non-positive values. This means the function is linear for positive inputs and flat (zero gradient) for negative inputs.

This function is particularly useful in deep learning models as it introduces non-linearity while being computationally efficient, helping to capture complex patterns in the data.
"""


# ==== Code ====
def relu(z: float) -> float:
    # Your code here
    return max(0, z)


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"z": 0}
    print(inp)
    out = relu(**inp)
    print(f"Output: {out}")
    exp = 0
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"z": 1}
    print(inp)
    out = relu(**inp)
    print(f"Output: {out}")
    exp = 1
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 2:")
    inp = {"z": -1}
    print(inp)
    out = relu(**inp)
    print(f"Output: {out}")
    exp = 0
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
