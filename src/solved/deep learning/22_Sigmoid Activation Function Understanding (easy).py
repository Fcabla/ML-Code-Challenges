"""
problem_id: 22
Category: deep learning
Difficulty: easy
URL: https://www.deep-ml.com/problem/Sigmoid%20Activation%20Function%20Understanding
Page: 1

==== Title ====
Sigmoid Activation Function Understanding (easy)

==== Description ====
Write a Python function that computes the output of the sigmoid activation function given an input value z. The function should return the output rounded to four decimal places.

==== Example ====
Example:
        input: z = 0
        output: 0.5
        reasoning: The sigmoid function is defined as σ(z) = 1 / (1 + exp(-z)). For z = 0, exp(-0) = 1, hence the output is 1 / (1 + 1) = 0.5.

==== Learn More ====
Understanding the Sigmoid Activation Function

The sigmoid activation function is crucial in neural networks, especially for binary classification tasks. It maps any real-valued number into the (0, 1) interval, making it useful for modeling probability as an output.

Mathematical Definition

The sigmoid function is mathematically defined as:

\[
\sigma(z) = \frac{1}{1 + e^{-z}}
\]

Where \(z\) is the input to the function.

Characteristics

Output Range: The output is always between 0 and 1.
Shape: It has an "S" shaped curve.
Gradient: The function's gradient is highest near \(z = 0\) and decreases toward either end of the z-axis.


This function is particularly useful for turning logits (raw prediction values) into probabilities in binary classification models.
"""

# ==== Code ====
import math


def sigmoid(z: float) -> float:
    # Your code here
    # σ(z) = 1 / (1 + exp(-z))
    result = 1 / (1 + math.exp(-z))
    return round(result, 4)


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"z": 0}
    print(inp)
    out = sigmoid(**inp)
    print(f"Output: {out}")
    exp = 0.5
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"z": 1}
    print(inp)
    out = sigmoid(**inp)
    print(f"Output: {out}")
    exp = 0.7311
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {"z": -1}
    print(inp)
    out = sigmoid(**inp)
    print(f"Output: {out}")
    exp = 0.2689
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
