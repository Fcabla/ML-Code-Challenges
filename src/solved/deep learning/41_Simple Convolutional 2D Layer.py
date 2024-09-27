"""
problem_id: 41
Category: deep learning
Difficulty: medium
URL: https://www.deep-ml.com/problem/Simple%20Convolutional%202D%20Layer
Page: 2

==== Title ====
Simple Convolutional 2D Layer

==== Description ====
In this problem, you need to implement a 2D convolutional layer in Python. This function will process an input matrix using a specified convolutional kernel, padding, and stride.

==== Example ====
Example:
import numpy as np

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)
# Expected Output:
# [[  3.   9.]
#  [ 11.  17.]]

==== Learn More ====
Simple Convolutional 2D Layer

Convolutional layer is widely used in Computer Vision tasks. Here is crucial parameters:

Parameters:


input_matrix: is a 2D NumPy array representing the input data, such as an image. Each element in this array corresponds to a pixel or a feature value in the input space. The dimensions of the input matrix are typically denoted as ${height, width}$.


kernel: is another 2D NumPy array representing the convolutional filter. The kernel is smaller than the input matrix and slides over it to perform the convolution operation. Each element in the kernel represents a weight that modifies the input as it is convolved over it. The kernel size is denoted as ${kernel}$_${height}$, ${kernel}$_${width}$.


padding: is an integer specifying the number of rows and columns of zeros added around the input matrix. Padding helps control the spatial dimensions of the output. It can be used to maintain the same output size as the input or to allow the kernel to process edge elements in the input matrix more effectively.


stride: is an integer representing the number of steps the kernel moves across the input matrix for each convolution operation. A stride greater than one results in a smaller output size, as the kernel skips over some elements.


Implementation:


Padding the Input: The input matrix is padded with zeros based on the `padding` value. This increases the input size, allowing the kernel to cover elements at the borders and corners that would otherwise be skipped.


Calculating Output Dimensions:The height and width of the output matrix are calculated using the formula:
    \[
     \text{output}_\text{height} = \left(\frac{\text{input}_{\text{height, padded}} - \text{kernel}_\text{height}}{\text{stride}}\right) + 1
     \]
     \[
     \text{output}_\text{width} = \left(\frac{\text{input}_\text{width, padded} - \text{kernel}_\text{width}}{\text{stride}}\right) + 1
     \]

Performing Convolution:

A nested loop iterates over each position where the kernel can be placed on the padded input matrix.
At each position, a region of the input matrix the same size as the kernel is selected.
Element-wise multiplication between the kernel and the input region is performed, followed by summation to produce a single value. This value is stored in the corresponding position in the output matrix.


Output: The function returns the output matrix, which contains the results of the convolution operation applied across the entire input.
"""

# ==== Code ====
import numpy as np


def simple_conv2d(
    input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int
):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape
    # Your code here
    # Pad the array (add 0s around the matrix)
    padded_input = np.pad(input_matrix, padding)
    input_height_padded, input_width_padded = padded_input.shape
    # Calculate the dimensions of the resulting matrix
    out_height = ((input_height_padded) - kernel_height) // stride + 1  # /stride)+1
    out_width = ((input_width_padded) - kernel_width) // stride + 1  # /stride)+1
    output_matrix = np.zeros((out_height, out_width))
    # Start convolution. Pass the kernel through the padded input
    for i in range(out_height):
        for j in range(out_width):
            region = padded_input[
                i * stride : i * stride + kernel_height,
                j * stride : j * stride + kernel_width,
            ]
            output_matrix[i, j] = np.sum(region * kernel)
    return output_matrix


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {
        "input_matrix": np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0, 25.0],
            ]
        ),
        "kernel": np.array(
            [
                [1.0, 2.0],
                [3.0, -1.0],
            ]
        ),
        "padding": 0,
        "stride": 1,
    }
    print(inp)
    out = simple_conv2d(**inp)
    print(f"Output: {out}")
    exp = np.array(
        [
            [16.0, 21.0, 26.0, 31.0],
            [41.0, 46.0, 51.0, 56.0],
            [66.0, 71.0, 76.0, 81.0],
            [91.0, 96.0, 101.0, 106.0],
        ]
    )
    print(f"Expected: {exp}")
    # print("Accepted" if out == exp else "Error")
    print("Accepted" if np.array_equal(out, exp) else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {
        "input_matrix": np.array(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0],
                [6.0, 7.0, 8.0, 9.0, 10.0],
                [11.0, 12.0, 13.0, 14.0, 15.0],
                [16.0, 17.0, 18.0, 19.0, 20.0],
                [21.0, 22.0, 23.0, 24.0, 25.0],
            ]
        ),
        "kernel": np.array(
            [
                [0.5, 3.2],
                [1.0, -1.0],
            ]
        ),
        "padding": 2,
        "stride": 2,
    }
    print(inp)
    out = simple_conv2d(**inp)
    print(f"Output: {out}")
    exp = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 5.9, 13.3, 12.5],
            [0.0, 42.9, 50.3, 27.5],
            [0.0, 80.9, 88.3, 12.5],
        ]
    )
    print(f"Expected: {exp}")
    # print("Accepted" if out == exp else "Error")
    # print("Accepted" if np.array_equal(out, exp) else "Error")
    print("Accepted" if np.allclose(out, exp) else "Error")
    print("---")
