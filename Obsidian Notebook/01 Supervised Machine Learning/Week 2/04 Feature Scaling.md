- When the range of values for a feature are large, then the weight or `w` of the parameter will be relatively small
- Vice versa for when the range is small
- When the ranges of values differ so much, it can cause gradient descent to run more slowly, so rescaling can fix this!
- We can rescale by getting `x` and dividing by the maximum
	 ![[Pasted image 20221105100810.png|center]]
	- The changes the scale to `[0.0 <= x <= 1.0]`
- **Mean Normalization** is using equation to plot the points near the origin of the graph
- **Z-Score Normalization** is using the standard deviation to plot the points
	![[Pasted image 20221105114158.png|center]]

```py
# recall that j in the column/feature

# calculating mean of a feature
µ[j] = 0
for i in range(m):
	µ[j] += x[i][j]
µ[j] = (µ[j] / m)

# calculating standard deviation of a feature
σ[j] = 0
for i in range(m):
	σ[j] += (x[i][j] - µ[j])**2
σ[j] = (σ[j] / m)

# z-score normalization
x[i][j] = ( x[i][j] - µ[j] ) / σ[j]
```
- This can be done easily using NumPy
```py
# find the mean of each column/feature
mu = np.mean(X, axis=0)

# find the standard deviation of each column/feature
sigma = np.std(X, axis=0)

# element-wise subtraction
X_norm = (X - mu) / sigma
```

![[Pasted image 20221105115646.png|center]]

- If a range is too large, or too small, rescaling is the answer

