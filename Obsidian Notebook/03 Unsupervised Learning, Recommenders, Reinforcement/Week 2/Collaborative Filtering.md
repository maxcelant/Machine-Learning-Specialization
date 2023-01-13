- Within this system, we have some number of users and items
- `n_u` = number of users
- `n_m` = number of items
- `r(i,j) = 1`, if user `j` has rated movie `i`
- `r(i,j) = 0`, if user `j` has rated movie `i`
- `y(i,j)` = rating given by user `j` to movie `i` (only if `r(i,j) = 1` )

![[Pasted image 20230103112335.png|center]]

- For example `y(3,2) = 4` because *Cute puppies of love* was given 4 stars by Bob
- `r(3,1) = 0` because *Cute puppies of love* is not rated by Alice

### Using Per-Item Features
- What if we had features of each movies?
- `n` is the number of features
- `X[i]` = feature vector for movie `i`
- `w[j]`, `b[j]` = parameters for user `j`

![[Pasted image 20230103112739.png|center]]

- For instance, Love at last aka `X[i]` would have features `[0.9 0]`
- For Alice: Predict rating for movie `i` as: `w * X[i] + b`
	- Lets say: `w[Alice] = [5 0]` and `b[Alice] = 0`
	- Also, for Cute puppies of love: `x[Cute puppies of love] = [0.99 0]`
	- Hence our prediction will be: `w[Alice] * x[Cute puppies of love] + b[Alice]`
	- or `w[1] * x[3] + b[1] = 4.95`
- Realize that the `w`eights and `b`ias of each user will be different
- To abstract this notation:
	- For user `j`: Predict user `j`'s rating for movie `i` as `w[j] * x[i] + b[j]`

### Cost Function
![[Pasted image 20230103115908.png|center]]

```python
cost = 0
for j in range(users):
	for i in range(movies):
		if r(i,j) == 1:
			cost += (w[j] * x[i] + b[j] - y[i][j])**2
	cost /= 2
```

### Collaborative Filtering Algorithm
- What if we are not given the feature vector for the movies?
- We can try to calculate it by looking at the outputs given by the users
![[Pasted image 20230103120323.png|center]]
- Based on the users inputs, since the users Alice and Bob (according to their weights) really like Romance movies (`[5 0]`), and they rated this one a 5/5, we can rightfully guess that the feature vector for *Love at Last* will be `[1 0]` with `1`being romance feature and `0` being action feature
![[Pasted image 20230103120715.png|center]]
- For all the users that have rated movie `i`, minimize the prediction calculated - the actual movie rating given by the user
- To learn all the features given, we sum over all features
![[Pasted image 20230103120919.png|center]]

```python
for i in range(movies):
	for j in range(users):
		if r(i,j) == 1:
			cost += (w[j] * x[i] + b[j] - y[i][j])**2
	cost /= 2
```
- What you may notice is that this looks very similar to the cost function for the user weights. Indeed, we actually can combine these two into a single equation
![[Pasted image 20230103121152.png|center]]

### Gradient Descent
![[Pasted image 20230103121333.png|center]]
- `x` now is also a parameter

### Binary Labels: Favorites, Likes, Clicks
- Using `1` to denote that the user liked, viewed or purchased something
- Using `0` to denote they didn't like it or didn't view it long enough
- Using `?` to denote the item was not yet shown to the user
- For binary labels: Preduct that the probability of `y(i,j) = 1` is given by `g(w[j] * x[i] + b[j])`
	- Where `g(z) = 1/1+e^-z`

### Mean Normalization
- Mean Normalization will help the algorithm make better predictions for users who have not yet rated any movies
- We take the average of all the ratings for each movie (call this `mu`) and subtract that amount from all the users ratings of that movie

![[Pasted image 20230106115850.png|center]]

- For user `j` on movie `i` predict:
	- `w[j] * x[i] + b[j] + mu[i]`
- For a new user (Eve),
	- `w = [0, 0] b = 0` - > `w[Eve] + X[Movie 1] + b[Eve] + mu[Movie 1] = 2.5`

### TensorFlow Implementation
- Called **Auto Diff / Auto Grad**
```python
w = tf.Variable(3.0) # Tf.variables are the parameters we want to optimize
x = 1.0
y = 1.0 # target value
alpha = 0.01

iterations = 30
for iter in range(iterations):

	# Use TensorFlow's Gradient tape to record the steps used to compute
	# the cost J to enable auto differentiation
	with tf.GradientTape() as tape:
		y_hat = w*x
		costJ = (y_hat - y)**2
	
	# Use the gradient tape to calculate the gradients 
	# of the cost with respect to the parameter w
	[dJdw] = tape.gradient(costJ, [w])
	
	# Run one step of gradient descent by updating
	# the value of w to reduce the cost
	w.assign_add(-alpha * dJdw)
```

- Implementation for **Recommender Systems**

```python
optimizer = keras.optimizer.Adam(learning_rate=1e-1)

iterations = 200
for iter in range(iterations):
	# Record the operations used to compute the cost
	with tf.GradientTape() as tape:
		cost_value = cofiCostFuncV(X, W, b, Ynorm, R, num_users, num_movies, lmbda)
	# Retrieve the gradients of the trainable variables
	# with respect to the loss
	grads = tape.gradient( cost_value, [X, W, b])

	# Tun one step of gradient descent by updating the 
	# value of the variables to minimize the loss
	optimizer.apply_gradients( zip(grads, [X, W, b]))
```

![[Pasted image 20230106121605.png|center]]

### Finding Related Items
- Given a movie 'Top Gun' with features `x[0] to x[n]` (a feature being romance, action, comedy, etc)
- Find an movie `k` with features similar to that of Top Gun
![[Pasted image 20230106122150.png|center]]
```python
diff = 0
for l in range(n): # n features (action, romance, etc)
	# i and k are two different movies
	diff += X[k][l] - X[i][l] 
```
