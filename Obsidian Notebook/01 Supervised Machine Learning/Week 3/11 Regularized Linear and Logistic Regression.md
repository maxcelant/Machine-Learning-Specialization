- The Gradient Descent algorithm changes slightly with regularization
![[Pasted image 20221107122148.png|center]]


#### Computing Cost Function with Regularization
![[Pasted image 20221107123646.png|center]]
```py
def compute_cost(X, y, w, b, lmbda):
	m, n = X.shape
	cost = 0
	for i in range(m):
		z = np.dot(w, X[i]) + b
		y_hat = 1 / (1 + np.exp(-z))
		cost += -y[i] * np.log(y_hat) - (1 - y[i]) * np.log(1 - y_hat)
	cost /= m

	reg_cost = 0
	for j in range(n):
		reg_cost += (w[j])**2
	reg_cost = (lmbda / (2*m)) * reg_cost
	total_cost = cost + reg_cost
	return total_cost
```

#### Computing Gradient with Regularization
![[Pasted image 20221107123618.png|center]]

```py
def calculate_gradient(X, y, w, b, lambda):
	
	dj_dw = np.zeros((n,))
	dj_db = 0
	m, n = X.shape
	
	for i in range(m):
		z = np.dot(w, X[i]) + b          # calculate learning algorithm
		y_hat = 1 / 1 + np.exp(-z)       # calculate sigmoid function
		cost = y_hat - y[i]            # calculate cost
		for j in range(n):               # calculate it for each feature
			dj_dw[j] += (cost * X[i,j])
		dj_db += cost
	
	dj_dw /= m
	dj_db /= m

	# regularization
	for j in range(n):
		dj_dw[j] += (lambda / m) * w[j]
	
	return dj_dw, dj_db
```