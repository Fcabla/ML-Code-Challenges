"""
problem_id: 26
Category: deep learning
Difficulty: medium
URL: https://www.deep-ml.com/problem/Implementing%20Basic%20Autograd%20Operations
Page: 1

==== Title ====
Implementing Basic Autograd Operations (medium)

==== Description ====
Special thanks to Andrej Karpathy for making a video about this, if you haven't already check out his videos on YouTube https://youtu.be/VMj-3S1tku0?si=gjlnFP4o3JRN9dTg. Write a Python class similar to the provided 'Value' class that implements the basic autograd operations: addition, multiplication, and ReLU activation. The class should handle scalar values and should correctly compute gradients for these operations through automatic differentiation.

==== Example ====
Example:
        a = Value(2)
        b = Value(-3)
        c = Value(10)
        d = a + b * c
        e = d.relu()
        e.backward()
        print(a, b, c, d, e)
        Output: Value(data=2, grad=0) Value(data=-3, grad=10) Value(data=10, grad=-3) Value(data=-28, grad=1) Value(data=0, grad=1)
        Explanation: The output reflects the forward computation and gradients after backpropagation. The ReLU on 'd' zeros out its output and gradient due to the negative data value.

==== Learn More ====
Understanding Mathematical Concepts in Autograd Operations

First off watch this: https://youtu.be/VMj-3S1tku0?si=gjlnFP4o3JRN9dTg

This task focuses on the implementation of basic automatic differentiation mechanisms for neural networks. The operations of addition, multiplication, and ReLU are fundamental to neural network computations and their training through backpropagation.

Mathematical Foundations

Addition (`__add__`):

Forward pass: For two scalar values \(a\) and \(b\), their sum \(s\) is simply \(s = a + b\).
Backward pass: The derivative of \(s\) with respect to both \(a\) and \(b\) is 1. Therefore, during backpropagation, the gradient of the output is passed directly to both inputs.


Multiplication (`__mul__`):

Forward pass: For two scalar values \(a\) and \(b\), their product \(p\) is \(p = a \times b\).
Backward pass: The gradient of \(p\) with respect to \(a\) is \(b\), and with respect to \(b\) is \(a\). This means that during backpropagation, each input's gradient is the product of the other input and the output's gradient.


ReLU Activation (`relu`):

Forward pass: The ReLU function is defined as \(R(x) = \max(0, x)\). This function outputs \(x\) if \(x\) is positive and 0 otherwise.
Backward pass: The derivative of the ReLU function is 1 for \(x > 0\) and 0 for \(x \leq 0\). Thus, the gradient is propagated through the function only if the input is positive; otherwise, it stops.



Conceptual Application in Neural Networks

Addition and Multiplication: These operations are ubiquitous in neural networks, forming the basis of computing weighted sums of inputs in the neurons.
ReLU Activation: Commonly used as an activation function in neural networks due to its simplicity and effectiveness in introducing non-linearity, making learning complex patterns possible.


Understanding these operations and their implications on gradient flow is crucial for designing and training effective neural network models. By implementing these from scratch, one gains deeper insights into the workings of more sophisticated deep learning libraries.
"""


# ==== Code ====
class Value:
    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        # Implement addition here
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        # Implement multiplication here
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def relu(self):
        # Implement ReLU here
        val = self.data if self.data > 0 else 0
        out = Value(val, (self,), "ReLU")

        def _backward():
            v = 1 if out.data > 0 else 0
            self.grad += v * out.grad

        out._backward = _backward
        return out

    def backward(self):
        node_list = []
        visited = set()

        def build_node_list(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_node_list(child)
                node_list.append(node)

        build_node_list(self)
        self.grad = 1
        for node in reversed(node_list):
            node._backward()


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    a = Value(2)
    b = Value(3)
    c = Value(10)
    d = a + b * c
    e = Value(7) * Value(2)
    f = e + d
    g = f.relu()
    print("Input:")
    print(a, b, c, d, e, f, g)
    g.backward()
    print("Expected:")
    print(
        "Value(data=2, grad=1), Value(data=3, grad=10), Value(data=10, grad=3), Value(data=32, grad=1), Value(data=14, grad=1), Value(data=46, grad=1),Value(data=46, grad=1)"
    )
    print("Output:")
    print(a, b, c, d, e, f, g)
