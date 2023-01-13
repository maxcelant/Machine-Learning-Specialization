- We can generalize the work we did in the last page to make it more robust
- We can perform each layer using a `dense(a_in, W, b)` function
- `W` is a 2D matrix with all the weights of each neuron in the layer
![[Pasted image 20221111095227.png]]
- `b` is simply 1D array with each bias for each neuron

```python
def dense(a_in, W, b):
	neurons = W.shape # number of neurons in this layer
	a_out = np.zeros(neurons)
	for j in range(neurons):
		w = W[:,j] # gets the elements in the same column
		z = np.dot(w, a_in) + b[j]
		a_out[j] = sigmoid(z)
	return a_out
```
- As you can see, it takes input from a previous layer `a_in` and returns the output for the next layer `a_out`
- These can be streamed together into a `sequential()` function
```python
def sequential(x):
	a1 = dense(x, W1, b1)
	a2 = dense(a1, W2, b2)
	a3 = dense(a2, W3, b3)
	a4 = dense(a3, W4, b4)
	f_x = a4
	return f_x
```