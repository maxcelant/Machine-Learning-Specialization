- Look at an unlabeled dataset of normal events and learn to raise red flags for unusual events
- Used for fraud detection, manufacturing, and monitoring computers in a data center
![[Pasted image 20221216123139.png|center]]

### Density Estimation
- Probability of X being seen in dataset at a certain part
- if `p(x_test) < ε (small number)`, then we raise a flag bc its an anomaly
- if `p(x_test) >= ε`, then were okay

### Gaussian (Normal) Distribution
![[Pasted image 20221216124755.png|center]]
- Probability of `x` is determined by a Gaussian with mean `µ`, variance `σ^2`
- Changing the value of `σ` will make the bell wider or taller
- Changing `µ` will move the center of the bell
![[Pasted image 20221216125839.png|center]]

<u>**Finding Mu**</u>
![[Pasted image 20230102152541.png|center]]

```python
for j in range(n): # for every feature of a sample
	for i in range(m): # sum that column
		mu[j] += X[i][j]
	mu[j] /= m
```
```python
mu = 1 / m * np.sum(X, axis = 0)
```

<u>**Finding Variance / Sigma**</u>
![[Pasted image 20230102152552.png|center]]
```python
for j in range(n): # for every feature of a sample
	for i in range(m): # get the variance
		var[j] += (X[i][j] - mu[j]) ** 2
	var[j] /= m
```

```python
var = 1 / m * np.sum((X - mu) ** 2, axis = 0)
```
![[Pasted image 20221216130326.png|center]]

### Anomaly Detection Algorithms
- The probability of `X[i]` can be broken down as..
	- `p(X[i]) = p(X[i][1]; µ_1, σ_1) + p(X[i][2]; µ_2, σ_2) + ... + p(X[i][n]; µ_n, σ_n)`
	- With the second value being the feature number for the example
	- Each feature has a specific `µ` and `σ` value, but every example shares these same values
```python
p_X = np.zeros((m,))
for i in range(m):
	for j in range(n): # number of features for training example
		p_X[i] *= p(x[i][j], µ[j], σ[j])
```
![[Pasted image 20221216132650.png|center]]
```python
def p(x, µ, σ):
	c = -(x - µ)**2 / (2 * σ**2)
	e = np.exp(c)
	return (1 / sqrt(2 * pi) * σ) * e
```



### Developing and Evaluating Anomaly Detection Systems
- Assume we have some labeled data, of anomalous `y = 1` and non-anomalous `y = 0` examples
- The cross-validation and test set will have anomalies labeled `y = 1` in them, but the majority will be normal examples
- Training: 6000 good engines
- CV: 2000 good engines (y = 0), 10 anomalous (y = 1)
- Test: 2000 good engines (y = 0), 10 anomalous (y = 1)

![[Pasted image 20221218171359.png|center]]

- Basically, use CV set to tweak the epsilon so that the training set correctly labels the anomalies

### Anomaly Detection vs Supervised Learning
- Use anomaly detection when you have a small amount of positive examples (1-20)
- Use Supervised Learning if you have a large number of positive and negative examples

### Choosing what features to use
- Very important for anomaly detection
- Use gaussian features
	- Using `plt.hist(x)`, this will create a histogram of that data
- If a feature isn't gaussian, sometimes `np.log(x)` can help, but also just try different things!
![[Pasted image 20221218180155.png|center]]
- You can try things like `plt.hist(x**2)` or `plt.hist(x**0.5)`, etc...
- Sometimes adding a new feature can help improve the performance of your ADA