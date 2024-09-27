"""
problem_id: 44
Category: deep learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Leaky%20ReLU
Page: 2

==== Title ====
Leaky ReLU Activation Function

==== Description ====
Write a Python function `leaky_relu` that implements the Leaky Rectified Linear Unit (Leaky ReLU) activation function. The function should take a float `z` as input and an optional float `alpha`, with a default value of 0.01, as the slope for negative inputs. The function should return the value after applying the Leaky ReLU function.

==== Example ====
Example:
print(leaky_relu(0))
# Output: 0

print(leaky_relu(1))
# Output: 1

print(leaky_relu(-1))
# Output: -0.01

print(leaky_relu(-2, alpha=0.1))
# Output: -0.2

==== Learn More ====
Understanding the Leaky ReLU Activation Function
The Leaky ReLU (Leaky Rectified Linear Unit) activation function is a variant of the ReLU function used in neural networks. It addresses the "dying ReLU" problem by allowing a small, non-zero gradient when the input is negative. This small slope for negative inputs helps keep the function active and helps prevent neurons from becoming inactive.
Mathematical Definition
The Leaky ReLU function is mathematically defined as:

\[
f(z) = \begin{cases}
z & \text{if } z > 0 \\
\alpha z & \text{if } z \leq 0
\end{cases}
\]

Where \(z\) is the input to the function and \(\alpha\) is a small positive constant, typically \(\alpha = 0.01\).
In this definition, the function returns \(z\) for positive values, and for negative values, it returns \(\alpha z\), allowing a small gradient to pass through.
Characteristics

Output Range: The output is in the range \((- \infty, \infty)\). Positive values are retained, while negative values are scaled by the factor \(\alpha\), allowing them to be slightly negative.
Shape: The function has a similar "L" shaped curve as ReLU, but with a small negative slope on the left side for negative \(z\), creating a small gradient for negative inputs.
Gradient: The gradient is 1 for positive values of \(z\) and \(\alpha\) for non-positive values. This allows the function to remain active even for negative inputs, unlike ReLU, where the gradient is zero for negative inputs.

This function is particularly useful in deep learning models as it mitigates the issue of "dead neurons" in ReLU by ensuring that neurons can still propagate a gradient even when the input is negative, helping to improve learning dynamics in the network.
"""


# ==== Code ====
def leaky_relu(z: float, alpha: float = 0.01) -> float | int:
    # Your code here
    return z if z > 0 else alpha * z


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"z": 5, "alpha": 0.01}
    print(inp)
    out = leaky_relu(**inp)
    print(f"Output: {out}")
    exp = 5
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"z": 1, "alpha": 0.01}
    print(inp)
    out = leaky_relu(**inp)
    print(f"Output: {out}")
    exp = 1
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {"z": -1, "alpha": 0.01}
    print(inp)
    out = leaky_relu(**inp)
    print(f"Output: {out}")
    exp = -0.01
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
