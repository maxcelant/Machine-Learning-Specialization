#### Get Data
```python
X,Y = load_coffee_data();
```

#### Normalize the Data
- Not a layer in our model!
- 'adapt' the data. This learns the mean and variance of the data set and saves the values internally.
- normalize the data
- always apply normalization
```python
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)
```

#### Tile/Copy Data
- We do this to increase training set and reduce the number of epochs needed
```python
Xt = np.tile(Xn,(1000,1)) # (200000, 2)
Yt= np.tile(Y,(1000,1))   # (200000, 1)
```

#### Tensorflow Model
```python
tf.random.set_seed(1234)  # applied to achieve consistent results
model = Sequential(
    [
        tf.keras.Input(shape=(2,)),
        Dense(3, activation='sigmoid', name = 'layer1'),
        Dense(1, activation='sigmoid', name = 'layer2')
     ]
)
```
- The `tf.keras.Input(shape=(2,)),` specifies the expected shape of the input. This allows Tensorflow to size the weights and bias parameters at this point. This is useful when exploring Tensorflow models. This statement can be omitted in practice and Tensorflow will size the network parameters when the input data is specified in the `model.fit` statement.
```python
model.summary()
```
- To get a quick summary of the model
-   In the first layer with 3 units, we expect W to have a size of (2,3) and 𝑏 should have 3 elements.
-   In the second layer with 1 unit, we expect W to have a size of (3,1) and 𝑏 should have 1 element.

#### Compile and Fit
```python
model.compile(
    loss = tf.keras.losses.BinaryCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01),
)
```
- The `model.compile` statement defines a loss function and specifies a compile optimization.
```python
model.fit(Xt,Yt,epochs=10)
```
- The `model.fit` statement runs gradient descent and fits the weights to the data.

```python
W1, b1 = model.get_layer("layer1").get_weights()
W2, b2 = model.get_layer("layer2").get_weights()
```
- After fitting, the weights have been updated

#### Making Predictions
- Output is probability of a good roast, to make a decision, one must apply the probability to a threshold. In this case, we will use 0.5
- The model is expecting one or more examples where examples are in the rows of matrix. In this case, we have two features so the matrix will be (m,2) where m is the number of examples
- Remeber, we must normalize our examples

```python
X_test = np.array([
    [200,13.9],  # postive example
    [200,17]])   # negative example
X_testn = norm_l(X_test)
predictions = model.predict(X_testn)
```

### Epochs and Batches
- In the `compile` statement above, the number of `epochs` was set to 10. This specifies that the entire data set should be applied during training 10 times. During training, you see output describing the progress of training that looks like this:

```
Epoch 1/10
6250/6250 [==============================] - 6s 910us/step - loss: 0.1782
```

The first line, `Epoch 1/10`, describes which epoch the model is currently running. For efficiency, the training data set is broken into 'batches'. The default size of a batch in Tensorflow is 32. There are 200000 examples in our expanded data set or 6250 batches. The notation on the 2nd line `6250/6250 [====` is describing which batch has been executed.

```python
# predictions = [0.963, 0.303]
yhat = np.zeros_like(predictions)
for i in range(len(predictions)):
    if predictions[i] >= 0.5:
        yhat[i] = 1
    else:
        yhat[i] = 0
# yhat = [1, 0]
```

- Can also be written like so:
```python
yhat = (predictions >= 0.5).astype(int)
print(f"decisions = \n{yhat}")
```