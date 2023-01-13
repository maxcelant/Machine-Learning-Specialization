- Recommend items to you based on features of user and item to find good match
![[Pasted image 20230107085637.png|center]]
- Vector size of movie and user could be different
- `X_u[j]` features of user `j`
- `X_m[i]` features of movie `i`
- We need to compute `V_u` and `V_m` for `X_u` and `X_m`
- `V` vectors need to be equal in size before they the dot product can be performed
- `V_u[j] • V_m[i]`

### Deep Learning for Content-Based Filtering
- Since we need to have both vectors to be the same size, we will use a neural network for each!
![[Pasted image 20230107090445.png|center]]
- Depending on the parameters of the neural network, you come up with different vectors
- So we train for the vectors to result in the lowest cost / error.

```python
J = 0
for i in range(m): # movies
	for j in range(u): # users
		if r(i, j) == 1:
			J += (V_u[j] - V_m[i] - y[i][j])**2
```

- To find movies similar to movie `i`:
	- `|| V_m[k] - V_m[i] ||^2`, the smaller the more similar it is

### Recommending for a large catalogue
- Use **Retrieval and Ranking**
- **Retrieval:**
	- Generate a large list of plausible item candidates.
		- For each of the last 10 movies watched by the user, find 10 most similar 
	- Combine items into list, removing duplicates and items already watched/purchased.
- **Ranking:**
	- Take the list and rank using learned model
	- Run it through the neural network and dot product it with the user features to see which movies rank better than others
	- Display ranked items to user

### TensorFlow Implementation
```python
user_NN = tf.keras.models.Sequential([
	tf.keras.layers.Dense(256, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(32),
])

item_NN = tf.keras.models.Sequential([
	tf.keras.layers.Dense(256, activation='relu'),
	tf.keras.layers.Dense(128, activation='relu'),
	tf.keras.layers.Dense(32),
])

# Create the user input and point to the base network
input_user = tf.keras.layers.Input(shape=(num_user_features))
vu = user_NN(input_user)
vu = tf.linalg.l2_normalize(vu, axis=1)

# Create the item input and point to the base network
input_item = tf.keras.layers.Input(shape=(num_item_features))
vm = item_NN(input_item)
vm = tf.linalg.l2_normalize(vm, axis=1)

# Measure the similarity of the two vector outputs
output = tf.keras.layers.Dot(axes=1)([vu, vm])

# Specify the input and output of the model
model = Model([input_user, input_item], output)

# Specify the cost function
cost_fn = tf.keras.losses.MeanSquaredError()

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss=cost_fn)
```