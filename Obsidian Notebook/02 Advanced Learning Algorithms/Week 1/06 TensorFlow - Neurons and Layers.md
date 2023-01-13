
#### Neuron without activation - Regression/Linear Model
```python
# training data
X_train = np.array([[1.0], [2.0]])
y_train = np.array([[300.0], [500.0]])

# same as y-hat = w • x + b
linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )

# returns weight and bias
w, b = linear_layer.get_weights()

# you can set custom weight and bias
set_w = np.array([[200]])
set_b = np.array([100])
linear_layer.set_weights([set_w, set_b])

# calc prediction
prediction_tf = linear_layer(X_train)
```

#### Neuron with Sigmoid Activation
```python
X_train = np.array([0., 1, 2, 3, 4, 5], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
Y_train = np.array([0,  0, 0, 1, 1, 1], dtype=np.float32).reshape(-1,1)  # 2-D Matrix
```
We can implement a logistic neuron by adding a sigmoid activation. The `Sequential` model is used to make a model with a plain stack of layers, where each layer has exactly one input tensor and one output tensor.
```python
model = Sequential(
    [
        tf.keras.layers.Dense(1, input_dim=1,  activation = 'sigmoid', name='layer1')
    ]
)
```
`model.summary()` shows the layers and number of parameters in the model. There is only one layer in this model and that layer has only one unit. The unit has two parameters, 𝑤 and 𝑏.
```python
model.summary()
```
We can get and set the parameters of this layer
```python
# get
logistic_layer = model.get_layer('layer1')
w,b = logistic_layer.get_weights()

# set
set_w = np.array([[2]])
set_b = np.array([-4.5])
logistic_layer.set_weights([set_w, set_b])
```

We can then produce an accurate prediction
```python
activation_1 = model.predict(X_train[0].reshape(1,1))
```
