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
