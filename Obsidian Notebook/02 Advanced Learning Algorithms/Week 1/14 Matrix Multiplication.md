- If you see a matrix, think of the **COLUMNS** of the matrix
- If you see the transpose of a matrix, think of the **ROWS** of the matrix
![[Pasted image 20221111114654.png|center]]

```python
W = np.array([[3, 5, 7, 9],
			  [4, 6, 8, 0]])

A = np.array([[1, 2],
			  [-1, -2],
			  [3, 4]])
A_T = A.T # Transpose

Z = np.matmul(A_T, W)
Z = A_T @ W # same thing as matmul
```