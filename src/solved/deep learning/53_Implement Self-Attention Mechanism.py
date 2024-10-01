"""
problem_id: 53
Category: deep learning
Difficulty: medium
URL: https://www.deep-ml.com/problem/Self-Attention%20Mechanism
Page: 3

==== Title ====
Implement Self-Attention Mechanism

==== Description ====
Task: Implement the Self-Attention Mechanism
Your task is to implement the self-attention mechanism, which is a fundamental component of transformer models, widely used in natural language processing and computer vision tasks. The self-attention mechanism allows a model to dynamically focus on different parts of the input sequence when generating a contextualized representation.
Your function should return the self-attention output as a numpy array.

==== Example ====
Example:
import numpy as np

X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)

print(output)

# Expected Output:
# [[1.660477 2.660477]
#  [2.339523 3.339523]]

==== Learn More ====
Self-Attention Mechanism


Self-Attention Mechanism
This document provides an overview of the self-attention mechanism, which is fundamental in transformer models for tasks like natural language processing and computer vision.
Practical Implementation

The self-attention mechanism allows the model to weigh the importance of different parts of the input sequence dynamically. This ability to assign varying levels of importance is key to capturing long-range dependencies, which is highly effective in tasks like language translation, text summarization, and machine vision.
The self-attention operation calculates attention scores for every input, determining how much focus to put on other inputs when generating a contextualized representation.

Mathematical Background

Self-Attention Calculation:

            Given an input sequence \(X\):
            \[
            Q = XW_Q, \quad K = XW_K, \quad V = XW_V
            \]
            Where \(Q\), \(K\), and \(V\) represent the Query, Key, and Value matrices respectively, and \(W_Q\), \(W_K\), \(W_V\) are learned weight matrices.

            The attention score is computed as:
            \[
            \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
            \]
            Where \(d_k\) is the dimension of the key vectors.
"""

# ==== Code ====
import numpy as np


def compute_qkv(X, W_q, W_k, W_v):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    return Q, K, V


def self_attention(Q, K, V):
    def softmax(x):
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    dk = len(K)
    # scores = (Q @ K.T) / np.sqrt(dk)
    # attention_weights = softmax(scores)
    # attention_output = attention_weights @ V
    attention_output = softmax(((Q @ K.T) / np.sqrt(dk))) @ V
    return np.round(attention_output, 8).tolist()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "X": np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]),
        "W_q": np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
        "W_k": np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
        "W_v": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    }
    print(inp)
    Q, K, V = compute_qkv(**inp)
    print(f"Output: {Q, K, V}")
    inp = {"Q": Q, "K": K, "V": V}
    print(inp)
    out = self_attention(**inp)
    print(f"Output: {out}")
    exp = [
        [8.0, 10.0, 12.0],
        [8.61987385, 10.61987385, 12.61987385],
        [7.38012615, 9.38012615, 11.38012615],
    ]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "X": np.array([[1, 1], [1, 0]]),
        "W_q": np.array([[1, 0], [0, 1]]),
        "W_k": np.array([[1, 0], [0, 1]]),
        "W_v": np.array([[1, 2], [3, 4]]),
    }
    print(inp)
    Q, K, V = compute_qkv(**inp)
    print(f"Output: {Q, K, V}")
    inp = {"Q": Q, "K": K, "V": V}
    print(inp)
    out = self_attention(**inp)
    print(f"Output: {out}")
    exp = [[3.00928465, 4.6790462], [2.5, 4.0]]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 3
    print("Test Case 3:")
    inp = {
        "X": np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]]),
        "W_q": np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
        "W_k": np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]]),
        "W_v": np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
    }
    print(inp)
    Q, K, V = compute_qkv(**inp)
    print(f"Output: {Q, K, V}")
    inp = {"Q": Q, "K": K, "V": V}
    print(inp)
    out = self_attention(**inp)
    print(f"Output: {out}")
    exp = [
        [8.0, 10.0, 12.0],
        [8.61987385, 10.61987385, 12.61987385],
        [7.38012615, 9.38012615, 11.38012615],
    ]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
