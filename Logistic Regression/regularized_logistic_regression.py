# Calculate Gradient
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

# Calculate Gradient Descent 
def gradient_descent(X, y, w_in, b_in, alpha, epochs):
	w = copy.deepcopy(w_in)
	b = b_in

	for i in range(epochs):
		# Calculate gradient and update the parameters
		dj_dw, dj_db = calculate_gradient(X, y, w, b)

		# Update parameters using w, b, alpha and gradient
		w = w - alpha * dj_dw
		b = b - alpha * dj_db
	
	return w, b
