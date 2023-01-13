- Instead of manually sending the data from `layer_1` to `layer_2`, etc.
- TensorFlow can string them together using `Sequential()`
```python
layer_1 = Dense(units=3, activation='sigmoid')
layer_2 = Dense(units=1, activation='sigmoid')
model = Sequential([layer_1, layer_2])

X = np.array([[200, 17],
			  [120,  5],
			  [425,  2],
			  [212, 18]])
y = np.array([1,0,0,1])
model.compile(...) # <- talk about this more next week
model.fit(X, y)

model.predit(x_new)
```
- Then all you have to do is run `.compile()` and `.fit()` to get your prediction
- You can predict using `.predict()` method
- We can re-write the `Sequential()` method like so:
```python
model = Sequential(
    [
        Dense(units=3, activation='sigmoid')
        Dense(units=1, activation='sigmoid')
    ]
)
```