"""
problem_id: 4
Category: linear algebra
Difficulty: easy
URL: https://www.deep-ml.com/problem/Calculate%20Mean%20by%20Row%20or%20Column
Page: 2

==== Title ====
Calculate Mean by Row or Column (easy)

==== Description ====
Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode. The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.

==== Example ====
Example1:
        input: matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'column'
        output: [4.0, 5.0, 6.0]
        reasoning: Calculating the mean of each column results in [(1+4+7)/3, (2+5+8)/3, (3+6+9)/3].

        Example 2:
        input: matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'row'
        output: [2.0, 5.0, 8.0]
        reasoning: Calculating the mean of each row results in [(1+2+3)/3, (4+5+6)/3, (7+8+9)/3].

==== Learn More ====
Calculate Mean by Row or Column

Calculating the mean of a matrix by row or column involves averaging the elements across the specified dimension. This operation provides insights into the distribution of values within the dataset, useful for data normalization and scaling.

 Row Mean
The mean of a row is computed by summing all elements in the row and dividing by the number of elements. For row \(i\), the mean is:
\[
\mu_{\text{row } i} = \frac{1}{n} \sum_{j=1}^{n} a_{ij}
\]
where \(a_{ij}\) is the matrix element in the \(i^{th}\) row and \(j^{th}\) column, and \(n\) is the total number of columns.

 Column Mean
Similarly, the mean of a column is found by summing all elements in the column and dividing by the number of elements. For column \(j\), the mean is:
\[
\mu_{\text{column } j} = \frac{1}{m} \sum_{i=1}^{m} a_{ij}
\]
where \(m\) is the total number of rows.

This mathematical formulation helps in understanding how data is aggregated across different dimensions, a critical step in various data preprocessing techniques.
"""


# ==== Code ====
def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    means = []
    if mode == "row":
        for row in matrix:
            means.append(sum(row) / len(row))

    if mode == "column":
        # Initialize means vector
        for _ in range(len(matrix[0])):
            means.append(0)
        for row in matrix:
            for i in range(len(row)):
                means[i] += row[i]
        for i in range(len(means)):
            means[i] = means[i] / len(matrix)

    return means


# ==== Test cases ====
if __name__ == "__main__":
    # Test case 1
    print("Test Case 1:")
    inp = {"matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "mode": "column"}
    print(inp)
    out = calculate_matrix_mean(**inp)
    print(f"Output: {out}")
    exp = [4.0, 5.0, 6.0]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")

    # Test case 2
    print("Test Case 2:")
    inp = {"matrix": [[1, 2, 3], [4, 5, 6], [7, 8, 9]], "mode": "row"}
    print(inp)
    out = calculate_matrix_mean(**inp)
    print(f"Output: {out}")
    exp = [2.0, 5.0, 8.0]
    print(f"Expected: {exp}")
    print("Accepted" if out == exp else "Error")
    print("---")
