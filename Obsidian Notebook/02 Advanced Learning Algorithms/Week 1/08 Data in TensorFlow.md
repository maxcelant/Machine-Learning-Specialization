**Why does NumPy use the double bracket in its arrays?**

This denotes a 1x2 matrix (Row Vector). TensorFlow **DOES** use this method
```python
x = np.array([[200, 17]]) # 1x2
```

This denotes a 2x1 matrix (Column Vector). TensorFlow **DOES** use this method
```python
x = np.array([[200]. [17]]) # 2x1
```

This results in a 1D vector, it has no rows or columns. TensorFlow does **NOT** use this method
```python
x = np.array([200, 17])
```

Think of a Tensor as a way of representing a matrix
This is a 1x1 tensor, which stores 1 value
```python
tf.Tensor([[0.8]], shape=(1,1), dtype=float32)
```



