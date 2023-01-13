- Using matrix multiplication, we can quickly perform the `A_out` of the entire layer
- Vectorizing the implementation
```python
X = np.array([[200, 17]])
W = np.array([[1, -3, 5],
			  [-2, 4, -6],
			  [-1, 1, 2]])

B = np.array([[-1, 1, 2]])

def dense(A_in, W, B):
	Z = np.matmul(A_in, W) + B
	A_out = sigmoid(Z)
	return A_out
```